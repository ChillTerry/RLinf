# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.modules.value_head import ValueHead
from rlinf.models.embodiment.uninavid.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from rlinf.models.embodiment.uninavid.mm_utils import tokenizer_image_token
from rlinf.models.embodiment.uninavid.model.uninavid_arch import (
    build_navigation_visual_tokens,
    process_grid,
    update_online_nav_cache,
)
from rlinf.models.embodiment.uninavid.nav_rollout import (
    UniNaVidNavCache,
    build_action_token_mask,
    build_action_token_slot_ids,
    build_navigation_prompt,
    count_parsed_action_chars,
    count_response_alpha_chars,
    empty_rollout_metadata,
    episode_id_from_obs,
    get_slot_cache,
    parse_uninavid_action_names,
    parse_uninavid_actions,
    select_slot_rgb_frames,
)
from rlinf.utils.utils import compute_logprobs_from_logits


class UniNaVidForActionPrediction(nn.Module, BasePolicy):
    """
    SFT-only wrapper around Uni-NaVid to fit RLinf's embodied policy interface.
    """

    _UNINAVID_FSDP_WRAP_NAMES = (
        "uninavid_vision_tower",
        "uninavid_mm_projector",
        "uninavid_lm_head",
    )

    def __init__(
        self,
        *,
        tokenizer,
        model,
        image_processor,
        torch_dtype: torch.dtype | None = None,
        cfg=None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.torch_dtype = torch_dtype
        self.cfg = cfg
        self._nav_caches: dict[int, UniNaVidNavCache] = {}
        self._response_token_stats_written = 0
        self._response_token_stats_steps: dict[int, tuple[int, int]] = {}

        if self._cfg_get(cfg, "add_value_head", default=False):
            self.value_head = ValueHead(
                input_dim=self._get_hidden_size_for_value_head(),
                hidden_sizes=tuple(
                    self._cfg_get(
                        cfg,
                        "value_head_hidden_sizes",
                        default=(512, 128),
                    )
                ),
                output_dim=1,
                activation=self._cfg_get(
                    cfg,
                    "value_head_activation",
                    default="relu",
                ),
                bias_last=self._cfg_get(
                    cfg,
                    "value_head_bias_last",
                    default=False,
                ),
            )
            if torch_dtype is not None:
                self.value_head = self.value_head.to(dtype=torch_dtype)

        self._initialize_fsdp_wrap_metadata()

    @property
    def _no_split_modules(self) -> list[str] | None:
        no_split_modules = getattr(self.model, "_no_split_modules", None)
        if no_split_modules is None:
            inner_model = getattr(self.model, "model", None)
            no_split_modules = getattr(inner_model, "_no_split_modules", None)
        if no_split_modules is None:
            return None
        return list(no_split_modules)

    @property
    def _no_split_names(self) -> list[str]:
        return list(self._UNINAVID_FSDP_WRAP_NAMES)

    @property
    def num_action_chunks(self) -> int:
        return self._get_num_action_chunks(mode="train")

    @classmethod
    def from_pretrained(
        cls,
        *,
        cfg,
        torch_dtype: torch.dtype | None = None,
    ) -> "UniNaVidForActionPrediction":
        from transformers import AutoConfig, AutoTokenizer

        from rlinf.models.embodiment.uninavid import conversation as conversation_lib
        from rlinf.models.embodiment.uninavid.model import LlavaLlamaAttForCausalLM

        model_name_or_path = cls._cfg_get(
            cfg,
            "model_name_or_path",
            "model_path",
            "pretrained_model_name_or_path",
            required=True,
        )
        cache_dir = cls._cfg_get(cfg, "cache_dir", default=None)
        model_max_length = int(cls._cfg_get(cfg, "model_max_length", default=512))
        dtype = cls._resolve_torch_dtype(torch_dtype)

        config = AutoConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )
        cls._apply_rope_scaling(config, model_max_length)
        cls._apply_multimodal_config_overrides(config, cfg)

        model = LlavaLlamaAttForCausalLM.from_pretrained(
            model_name_or_path,
            config=config,
            cache_dir=cache_dir,
            low_cpu_mem_usage=True,
            torch_dtype=dtype,
        )
        model.config.use_cache = False

        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            model_max_length=model_max_length,
            padding_side="right",
            use_fast=False,
        )

        version = cls._cfg_get(cfg, "version", default="v0")
        if version == "v0":
            if tokenizer.pad_token is None:
                cls._smart_tokenizer_and_embedding_resize(
                    special_tokens_dict={"pad_token": "[PAD]"},
                    tokenizer=tokenizer,
                    model=model,
                )
        elif version == "v0.5":
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.pad_token = tokenizer.unk_token
            conversation_lib.default_conversation = conversation_lib.conv_templates.get(
                version,
                conversation_lib.conv_templates["vicuna_v1"],
            )

        model_args = cls._build_model_args(cfg, model_name_or_path)
        image_processor = None
        if model_args.vision_tower is not None:
            model.get_model().initialize_vision_modules(
                model_args=model_args,
                fsdp=cls._cfg_get(cfg, "fsdp", default=None),
                max_token=model_max_length,
            )

            vision_tower = model.get_vision_tower()
            device = cls._cfg_get(
                cfg,
                "device",
                default="cuda" if torch.cuda.is_available() else "cpu",
            )
            vision_tower.to(dtype=dtype, device=device)
            image_processor = vision_tower.image_processor

            model.config.image_aspect_ratio = cls._cfg_get(
                cfg,
                "image_aspect_ratio",
                default="square",
            )
            model.config.image_grid_pinpoints = cls._cfg_get(
                cfg,
                "image_grid_pinpoints",
                default=None,
            )
            model.config.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
            model.config.freeze_mm_mlp_adapter = cls._cfg_get(
                cfg,
                "freeze_mm_mlp_adapter",
                default=False,
            )
            model.config.mm_use_im_start_end = model_args.mm_use_im_start_end
            model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token

            cls._apply_upstream_trainability(
                model=model,
                vision_tower=vision_tower,
                freeze_backbone=model_args.freeze_backbone,
                tune_mm_mlp_adapter=model_args.tune_mm_mlp_adapter,
                freeze_mm_mlp_adapter=cls._cfg_get(
                    cfg,
                    "freeze_mm_mlp_adapter",
                    default=False,
                ),
                tune_vision_encoder=cls._cfg_get(
                    cfg,
                    "tune_vision_encoder",
                    default=False,
                ),
            )

            model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

        return cls(
            tokenizer=tokenizer,
            model=model,
            image_processor=image_processor,
            torch_dtype=dtype,
            cfg=cfg,
        )

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.SFT:
            return self.sft_forward(**kwargs)
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        raise NotImplementedError

    def sft_forward(self, data: dict[str, Any] | None = None, **kwargs):
        if data is None:
            data = kwargs.get("data")
        if data is None and "input_ids" in kwargs:
            data = kwargs

        device = self._get_device()
        input_ids = data["input_ids"].to(device=device)
        attention_mask = data["attention_mask"].to(device=device)
        labels = data["labels"].to(device=device)
        images = self._move_images(data.get("images"), device=device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            images=images,
            prompts=data.get("prompts"),
            use_cache=False,
            return_dict=True,
        )
        loss = getattr(outputs, "loss", None)
        return loss

    def default_forward(
        self,
        forward_inputs: dict[str, torch.Tensor] | None = None,
        compute_logprobs: bool = True,
        compute_entropy: bool = False,
        compute_values: bool = False,
        **kwargs,
    ) -> dict[str, torch.Tensor | None]:
        prompt_inputs_embeds = forward_inputs["prompt_inputs_embeds"]
        prompt_attention_mask = forward_inputs["prompt_attention_mask"]
        response_ids = forward_inputs["response_ids"]
        response_mask = forward_inputs["response_mask"].to(torch.bool)

        logits_result = self._compute_logits_from_embeds(
            prompt_inputs_embeds=prompt_inputs_embeds,
            prompt_attention_mask=prompt_attention_mask,
            response_ids=response_ids,
            response_mask=response_mask,
            return_hidden_state=compute_values,
        )
        if compute_values:
            response_logits, final_hidden, prompt_len = logits_result
        else:
            response_logits = logits_result

        logprobs = None
        if compute_logprobs:
            logprobs = self._gather_masked_token_logprobs(
                logits=response_logits.float(),
                target=response_ids,
                mask=response_mask,
            )

        entropy = None
        if compute_entropy:
            response_logits = response_logits.float()
            probs = torch.softmax(response_logits, dim=-1)
            token_logprobs_all = torch.log_softmax(response_logits, dim=-1)
            entropy = -(probs * token_logprobs_all).sum(dim=-1, keepdim=True)
            entropy = entropy * response_mask.unsqueeze(-1).to(entropy.dtype)

        values = None
        if compute_values:
            values = self._compute_value_from_pre_response_hidden(
                final_hidden=final_hidden,
                prompt_len=prompt_len,
            )

        return {"logprobs": logprobs, "entropy": entropy, "values": values}

    def predict_action_batch(
        self,
        env_obs,
        calculate_logprobs: bool = False,
        calculate_values: bool = False,
        mode: str = "eval",
        **kwargs,
    ):
        num_action_chunks = self._get_num_action_chunks(
            mode=mode,
            override=kwargs.pop("num_action_chunks", None),
        )
        if mode == "train":
            return self._predict_train_batch(
                env_obs,
                kwargs,
                num_action_chunks=num_action_chunks,
                calculate_values=calculate_values,
            )

        return self._predict_eval_batch(
            env_obs,
            kwargs,
            num_action_chunks=num_action_chunks,
        )

    def _get_num_action_chunks(self, mode: str, override: int | None = None) -> int:
        if override is not None:
            return int(override)
        cfg = getattr(self, "cfg", None)
        if mode == "eval":
            eval_num_action_chunks = self._cfg_get(
                cfg,
                "eval_num_action_chunks",
                default=None,
            )
            if eval_num_action_chunks is not None:
                return int(eval_num_action_chunks)

        num_action_chunks = self._cfg_get(
            cfg,
            "num_action_chunks",
            default=4,
        )
        return int(num_action_chunks)

    def _predict_eval_batch(
        self,
        env_obs: dict[str, Any],
        generation_kwargs: dict[str, Any],
        *,
        num_action_chunks: int,
    ):
        output_texts, _, _, response_ids, _ = self._generate_batch_outputs(
            env_obs,
            generation_kwargs,
        )
        self._maybe_log_response_token_stats(
            env_obs=env_obs,
            output_texts=output_texts,
            response_mask=self._build_response_mask(response_ids),
            mode="eval",
            num_action_chunks=num_action_chunks,
        )
        action_chunks = []
        cfg = getattr(self, "cfg", None)
        log_eval_responses = bool(
            self._cfg_get(cfg, "log_eval_responses", default=False)
        )
        for slot_id, output_text in enumerate(output_texts):
            normalized_output_text = output_text.strip()
            parsed_actions = parse_uninavid_actions(
                normalized_output_text,
                num_action_chunks,
            )
            if log_eval_responses:
                parsed_names = parse_uninavid_action_names(
                    normalized_output_text,
                    num_action_chunks,
                )
                print(
                    f"[UNINAVID RESPONSE] slot={slot_id} "
                    f"parsed_actions={parsed_names!r} "
                    f"raw={normalized_output_text!r}",
                    flush=True,
                )
            action_chunks.append(parsed_actions)
        actions = torch.stack(action_chunks, dim=0)
        return actions, empty_rollout_metadata()

    def _predict_train_batch(
        self,
        env_obs: dict[str, Any],
        generation_kwargs: dict[str, Any],
        *,
        num_action_chunks: int,
        calculate_values: bool = False,
    ):
        (
            output_texts,
            prompt_inputs_embeds,
            prompt_attention_mask,
            response_ids,
            generated_scores,
        ) = self._generate_batch_outputs(
            env_obs,
            generation_kwargs,
            return_scores=True,
        )
        action_chunks = []
        action_token_masks = []
        action_token_slot_ids_list = []
        parsed_action_char_counts = []
        response_alpha_char_counts = []
        for output_text, response_id in zip(output_texts, response_ids):
            parsed_actions = parse_uninavid_actions(
                output_text.strip(),
                num_action_chunks,
            )
            action_chunks.append(parsed_actions)
            action_token_masks.append(
                torch.tensor(
                    build_action_token_mask(
                        output_text,
                        response_id,
                        num_action_chunks,
                    ),
                    dtype=torch.bool,
                    device=response_ids.device,
                )
            )
            action_token_slot_ids_list.append(
                torch.tensor(
                    build_action_token_slot_ids(
                        output_text,
                        response_id,
                        num_action_chunks,
                    ),
                    dtype=torch.long,
                    device=response_ids.device,
                )
            )
            parsed_action_char_counts.append(
                count_parsed_action_chars(output_text, num_action_chunks)
            )
            response_alpha_char_counts.append(count_response_alpha_chars(output_text))
        actions = torch.stack(action_chunks, dim=0)
        action_token_mask = torch.stack(action_token_masks, dim=0)
        action_token_slot_ids = torch.stack(action_token_slot_ids_list, dim=0)
        max_new_tokens = generation_kwargs.get("max_new_tokens")
        if max_new_tokens is None:
            response_len = int(response_ids.shape[1])
        else:
            response_len = int(max_new_tokens)
        cfg = getattr(self, "cfg", None)
        model_max_length = int(
            self._cfg_get(
                cfg,
                "model_max_length",
                default=getattr(self.model, "model_max_length", 0),
            )
        )
        if model_max_length <= 0:
            prompt_len = int(prompt_inputs_embeds.shape[1])
        else:
            prompt_len = model_max_length - response_len
            if prompt_len <= 0:
                raise ValueError(
                    "UniNaVid train metadata requires model_max_length > max_new_tokens."
                )
        drop_overlong_train_metadata = bool(
            self._cfg_get(cfg, "drop_overlong_train_metadata", default=False)
        )
        overlong_train_metadata_mask = None
        (
            prompt_inputs_embeds,
            prompt_attention_mask,
            overlong_train_metadata_mask,
        ) = self._left_pad_prompt_forward_inputs(
            prompt_inputs_embeds=prompt_inputs_embeds,
            prompt_attention_mask=prompt_attention_mask,
            target_len=prompt_len,
            mask_overlong=drop_overlong_train_metadata,
        )
        prev_logprobs, response_mask = self._compute_generation_score_logprobs(
            generated_scores=generated_scores,
            response_ids=response_ids,
        )
        response_ids, response_mask, prev_logprobs = (
            self._pad_response_inputs_with_logprobs(
                response_ids=response_ids,
                response_mask=response_mask,
                prev_logprobs=prev_logprobs,
                target_len=response_len,
            )
        )
        if action_token_mask.shape[1] < response_len:
            action_token_mask = torch.cat(
                [
                    action_token_mask,
                    torch.zeros(
                        (
                            action_token_mask.shape[0],
                            response_len - action_token_mask.shape[1],
                        ),
                        dtype=torch.bool,
                        device=action_token_mask.device,
                    ),
                ],
                dim=1,
            )
            action_token_slot_ids = torch.cat(
                [
                    action_token_slot_ids,
                    torch.full(
                        (
                            action_token_slot_ids.shape[0],
                            response_len - action_token_slot_ids.shape[1],
                        ),
                        -1,
                        dtype=torch.long,
                        device=action_token_slot_ids.device,
                    ),
                ],
                dim=1,
            )
        if action_token_mask.shape[1] != response_len:
            raise ValueError(
                "UniNaVid action token mask length must match response length."
            )
        if action_token_slot_ids.shape != action_token_mask.shape:
            raise ValueError(
                "UniNaVid action token slot ids must match response length."
            )
        if (
            overlong_train_metadata_mask is not None
            and overlong_train_metadata_mask.any()
        ):
            prev_logprobs = prev_logprobs.masked_fill(
                overlong_train_metadata_mask[:, None, None],
                0.0,
            )
            action_token_mask = action_token_mask.masked_fill(
                overlong_train_metadata_mask[:, None],
                False,
            )
            action_token_slot_ids = action_token_slot_ids.masked_fill(
                overlong_train_metadata_mask[:, None],
                -1,
            )
        prev_values = None
        if calculate_values:
            with torch.no_grad():
                prev_values = self._compute_values_from_forward_inputs(
                    prompt_inputs_embeds=prompt_inputs_embeds,
                    prompt_attention_mask=prompt_attention_mask,
                    response_ids=response_ids,
                    response_mask=response_mask,
                )
            if (
                overlong_train_metadata_mask is not None
                and overlong_train_metadata_mask.any()
            ):
                prev_values = prev_values.masked_fill(
                    overlong_train_metadata_mask[:, None],
                    0.0,
                )
        self._maybe_log_response_token_stats(
            env_obs=env_obs,
            output_texts=output_texts,
            response_mask=response_mask,
            mode="train",
            num_action_chunks=num_action_chunks,
        )
        metadata = {
            "prev_logprobs": prev_logprobs.detach(),
            "prev_values": prev_values.detach() if prev_values is not None else None,
            "forward_inputs": {
                "prompt_inputs_embeds": prompt_inputs_embeds.detach(),
                "prompt_attention_mask": prompt_attention_mask.detach(),
                "response_ids": response_ids.detach(),
                "response_mask": response_mask.detach(),
                "action_token_mask": action_token_mask.detach(),
                "action_token_slot_ids": action_token_slot_ids.detach(),
                "action": actions.reshape(actions.shape[0], -1).detach(),
                "parsed_action_char_count": torch.tensor(
                    parsed_action_char_counts,
                    dtype=torch.long,
                    device=response_ids.device,
                ),
                "response_alpha_char_count": torch.tensor(
                    response_alpha_char_counts,
                    dtype=torch.long,
                    device=response_ids.device,
                ),
            },
        }
        return actions, metadata

    def _predict_train_batch_cached(
        self,
        env_obs: dict[str, Any],
        generation_kwargs: dict[str, Any],
    ):
        return self._predict_train_batch(
            env_obs,
            generation_kwargs,
            num_action_chunks=self._get_num_action_chunks(mode="train"),
        )

    def _generate_batch_outputs(
        self,
        env_obs: dict[str, Any],
        generation_kwargs: dict[str, Any],
        *,
        return_scores: bool = False,
    ):
        batch_size = len(env_obs["task_descriptions"])
        prompts: list[str] = []
        embeds: list[torch.Tensor] = []

        missing_run_type = object()
        original_run_type = getattr(self.model.config, "run_type", missing_run_type)
        self.model.config.run_type = "eval"
        try:
            for slot_id in range(batch_size):
                episode_id = episode_id_from_obs(env_obs, slot_id)
                cache = get_slot_cache(
                    self._nav_caches,
                    slot_id=slot_id,
                    episode_id=episode_id,
                )
                instruction = env_obs["task_descriptions"][slot_id]
                navigation_prompt = build_navigation_prompt(instruction)
                prompts.append(
                    navigation_prompt.replace(DEFAULT_IMAGE_TOKEN, "").replace("\n", "")
                )
                full_input_ids = self._build_navigation_input_ids(navigation_prompt)
                input_ids = full_input_ids[0]

                rgb_frames = select_slot_rgb_frames(env_obs, slot_id)
                visual_features = self._encode_rgb_frames_for_slot(rgb_frames)
                current_tokens = self._update_slot_feature_cache(
                    cache,
                    visual_features,
                    new_frames=len(rgb_frames),
                )
                compress_type = getattr(self.model.config, "compress_type", None)
                nav_sizes = {"grid:2": 4, "grid:4": 16, "mean": 1}
                history_tokens, history_lengths = build_navigation_visual_tokens(
                    cache,
                    nav_sizes[compress_type],
                )
                embeds.append(
                    self._build_navigation_inputs_embeds(
                        input_ids,
                        history_tokens,
                        history_lengths,
                        current_tokens,
                    )
                )
            inputs_embeds, attention_mask = self._pad_navigation_embeds(embeds)
            self.model.update_prompt([[prompt] for prompt in prompts])
            generate_kwargs = dict(generation_kwargs)
            if return_scores:
                generate_kwargs["return_dict_in_generate"] = True
                generate_kwargs["output_scores"] = True
            outputs = self.model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=True,
                **generate_kwargs,
            )
            if return_scores:
                generated_scores = torch.stack(tuple(outputs.scores), dim=1).float()
                response_ids = outputs.sequences[:, 1 : 1 + generated_scores.shape[1]]
            else:
                generated_scores = None
                output_ids = (
                    outputs.sequences if hasattr(outputs, "sequences") else outputs
                )
                # HF generation starts from a synthetic one-token sequence when
                # inputs_embeds are provided without input_ids. Exclude that seed from
                # response-only rollout metadata and old-logprob recomputation.
                response_ids = output_ids[:, 1:]
            output_texts = self.tokenizer.batch_decode(
                response_ids,
                skip_special_tokens=True,
            )
            return (
                output_texts,
                inputs_embeds,
                attention_mask,
                response_ids,
                generated_scores,
            )
        finally:
            if original_run_type is missing_run_type:
                if hasattr(self.model.config, "run_type"):
                    delattr(self.model.config, "run_type")
            else:
                self.model.config.run_type = original_run_type

    def _maybe_log_response_token_stats(
        self,
        *,
        env_obs: dict[str, Any],
        output_texts: list[str],
        response_mask: torch.Tensor,
        mode: str,
        num_action_chunks: int,
    ) -> None:
        cfg = getattr(self, "cfg", None)
        if not bool(self._cfg_get(cfg, "log_response_token_stats", default=False)):
            return
        stats_path = self._cfg_get(
            cfg,
            "response_token_stats_path",
            default=None,
        )
        if not stats_path:
            return

        max_samples = int(
            self._cfg_get(cfg, "response_token_stats_max_samples", default=0)
        )
        if max_samples > 0 and self._response_token_stats_written >= max_samples:
            return

        rank = self._get_response_token_stats_rank()
        output_path = self._resolve_response_token_stats_path(str(stats_path), rank)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        global_step = getattr(self, "global_step", None)
        response_lengths = response_mask.to(torch.bool).sum(dim=1).detach().cpu()
        records = []
        for slot_id, output_text in enumerate(output_texts):
            if max_samples > 0 and self._response_token_stats_written >= max_samples:
                break

            episode_id = episode_id_from_obs(env_obs, slot_id)
            step = self._next_response_token_stats_step(slot_id, episode_id)
            parsed_actions = parse_uninavid_action_names(
                output_text,
                num_action_chunks,
            )
            records.append(
                {
                    "split": self._get_env_obs_list_value(
                        env_obs,
                        "split",
                        slot_id,
                    ),
                    "mode": mode,
                    "rank": rank,
                    "global_step": global_step,
                    "episode_id": episode_id,
                    "step": step,
                    "slot_id": slot_id,
                    "language": self._get_env_obs_list_value(
                        env_obs,
                        "languages",
                        slot_id,
                    ),
                    "response_token_len": int(response_lengths[slot_id].item()),
                    "parsed_actions": parsed_actions,
                    "parsed_action_count": len(parsed_actions),
                    "response_text": output_text,
                }
            )
            self._response_token_stats_written += 1

        if not records:
            return
        with output_path.open("a", encoding="utf-8") as file_obj:
            for record in records:
                file_obj.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _next_response_token_stats_step(self, slot_id: int, episode_id: int) -> int:
        previous = self._response_token_stats_steps.get(slot_id)
        if previous is None or previous[0] != episode_id:
            step = 0
        else:
            step = previous[1]
        self._response_token_stats_steps[slot_id] = (episode_id, step + 1)
        return step

    @staticmethod
    def _get_env_obs_list_value(
        env_obs: dict[str, Any],
        key: str,
        slot_id: int,
    ) -> Any:
        values = env_obs.get(key)
        if values is None:
            return None
        if isinstance(values, torch.Tensor):
            value = values[slot_id]
            if value.numel() == 1:
                return value.item()
            return value.detach().cpu().tolist()
        return values[slot_id]

    @staticmethod
    def _get_response_token_stats_rank() -> int:
        for env_key in ("RANK", "LOCAL_RANK"):
            value = os.environ.get(env_key)
            if value is not None:
                return int(value)
        return 0

    @staticmethod
    def _resolve_response_token_stats_path(path: str, rank: int) -> Path:
        if "{rank}" in path:
            return Path(path.format(rank=rank))
        output_path = Path(path)
        suffix = output_path.suffix
        stem = output_path.stem if suffix else output_path.name
        return output_path.with_name(f"{stem}_rank_{rank}{suffix}")

    def _build_navigation_input_ids(self, navigation_prompt: str) -> torch.Tensor:
        from rlinf.models.embodiment.uninavid import conversation as conversation_lib
        from rlinf.models.embodiment.uninavid.constants import (
            IAMGE_SEPARATOR,
            IMAGE_END_TOKEN,
            IMAGE_START_TOKEN,
            NAVIGATION_SPECIAL_TOKEN,
            VIDEO_END_SPECIAL_TOKEN,
            VIDEO_START_SPECIAL_TOKEN,
        )

        if self.model.config.mm_use_im_start_end:
            qs = (
                DEFAULT_IM_START_TOKEN
                + DEFAULT_IMAGE_TOKEN
                + DEFAULT_IM_END_TOKEN
                + "\n"
                + navigation_prompt.replace("<image>", "")
            )
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + navigation_prompt.replace("<image>", "")
        conv_mode = self._cfg_get(
            getattr(self, "cfg", None),
            "conv_mode",
            default="vicuna_v1",
        )
        conv = conversation_lib.conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        token_prompt = tokenizer_image_token(
            prompt,
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        )
        device = self._get_device()
        token_prompt = token_prompt.to(device=device)

        video_start = (
            self.tokenizer(
                VIDEO_START_SPECIAL_TOKEN,
                return_tensors="pt",
            )
            .input_ids[0][1:]
            .to(device)
        )
        image_separator = (
            self.tokenizer(
                IAMGE_SEPARATOR,
                return_tensors="pt",
            )
            .input_ids[0][1:]
            .to(device)
        )
        video_end = (
            self.tokenizer(
                VIDEO_END_SPECIAL_TOKEN,
                return_tensors="pt",
            )
            .input_ids[0][1:]
            .to(device)
        )
        image_start = (
            self.tokenizer(
                IMAGE_START_TOKEN,
                return_tensors="pt",
            )
            .input_ids[0][1:]
            .to(device)
        )
        image_end = (
            self.tokenizer(
                IMAGE_END_TOKEN,
                return_tensors="pt",
            )
            .input_ids[0][1:]
            .to(device)
        )
        navigation = (
            self.tokenizer(
                NAVIGATION_SPECIAL_TOKEN,
                return_tensors="pt",
            )
            .input_ids[0][1:]
            .to(device)
        )

        pieces: list[torch.Tensor] = []
        while True:
            indices = torch.where(token_prompt == IMAGE_TOKEN_INDEX)[0]
            if indices.numel() == 0:
                if token_prompt.numel() > 0:
                    pieces.append(token_prompt)
                break
            idx = indices[0]
            pieces.extend(
                [
                    token_prompt[:idx],
                    video_start,
                    image_separator,
                    token_prompt[idx : idx + 1],
                    video_end,
                    image_start,
                    image_end,
                    navigation,
                ]
            )
            token_prompt = token_prompt[idx + 1 :]

        nonempty_pieces = [piece for piece in pieces if piece.numel() > 0]
        return torch.cat(nonempty_pieces, dim=0).unsqueeze(0)

    def _build_navigation_inputs_embeds(
        self,
        input_ids: torch.Tensor,
        history_tokens: torch.Tensor,
        history_lengths: list[int],
        current_tokens: torch.Tensor,
    ) -> torch.Tensor:
        image_token_indices = torch.where(input_ids == IMAGE_TOKEN_INDEX)[0]
        if image_token_indices.numel() != 1:
            raise ValueError(
                "Uni-NaVid navigation prompt must contain exactly one image token."
            )

        image_token_start = int(image_token_indices[0].item())
        embed_tokens = self.model.get_model().embed_tokens
        pieces = [
            embed_tokens(input_ids[:image_token_start]),
        ]

        separator_token = embed_tokens(input_ids[image_token_start - 1, None])
        video_index = 0
        for idx, token_length in enumerate(history_lengths):
            pieces.append(history_tokens[video_index : video_index + token_length])
            video_index += token_length
            if idx != len(history_lengths) - 1:
                pieces.append(separator_token)

        pieces.append(
            embed_tokens(input_ids[image_token_start + 1 : image_token_start + 3])
        )
        pieces.append(current_tokens)
        pieces.append(embed_tokens(input_ids[image_token_start + 3 :]))
        return torch.cat(pieces, dim=0)

    def _encode_rgb_frames_for_slot(self, rgb_frames: list[Any]) -> torch.Tensor:
        images = self._preprocess_navigation_images(rgb_frames)[0]
        return self._encode_preprocessed_rgb_frames(images)

    def _encode_preprocessed_rgb_frames(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        vision_tower = self.model.get_vision_tower()
        visual_features = vision_tower(images)
        if (
            getattr(self.model.config, "mm_vision_select_feature", "patch") == "patch"
            and visual_features.shape[1] % 2 == 1
        ):
            visual_features = visual_features[:, 1:]
        return visual_features

    def _update_slot_feature_cache(
        self,
        cache: UniNaVidNavCache,
        visual_features: torch.Tensor,
        new_frames: int,
    ) -> torch.Tensor:
        compress_type = getattr(self.model.config, "compress_type", None)
        history_tokens = process_grid(
            visual_features,
            int(compress_type.split("grid:")[-1]),
        )
        history_tokens = self.model.get_model().mm_projector(history_tokens)
        update_online_nav_cache(cache, history_tokens, new_frames=new_frames)

        current_tokens = process_grid(visual_features[-1:], 8)
        current_tokens = self.model.get_model().mm_projector(current_tokens)[0]
        return current_tokens

    def _preprocess_navigation_images(
        self,
        rgb_frames: list[Any],
    ) -> list[torch.Tensor]:
        import numpy as np

        batch_image = np.asarray(rgb_frames)
        pixel_values = self.image_processor.preprocess(
            batch_image,
            return_tensors="pt",
        )["pixel_values"]
        if self.torch_dtype is None:
            pixel_values = pixel_values.to(device=self._get_device())
        else:
            pixel_values = pixel_values.to(
                device=self._get_device(),
                dtype=self.torch_dtype,
            )
        return [pixel_values]

    def _pad_navigation_embeds(
        self,
        embeds: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_len = max(embed.shape[0] for embed in embeds)
        hidden_size = embeds[0].shape[-1]
        batch = embeds[0].new_zeros((len(embeds), max_len, hidden_size))
        attention_mask = torch.zeros(
            (len(embeds), max_len),
            dtype=torch.long,
            device=embeds[0].device,
        )
        for idx, embed in enumerate(embeds):
            start = max_len - embed.shape[0]
            batch[idx, start:] = embed
            attention_mask[idx, start:] = 1
        return batch, attention_mask

    def _left_pad_prompt_forward_inputs(
        self,
        prompt_inputs_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        target_len: int,
        *,
        mask_overlong: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        prompt_token_mask = prompt_attention_mask.to(torch.bool)
        prompt_token_lengths = prompt_token_mask.sum(dim=1)
        overlong_mask = prompt_token_lengths.gt(target_len)
        if overlong_mask.any() and not mask_overlong:
            raise ValueError(
                "UniNaVid train metadata prompt length exceeds "
                "model_max_length - max_new_tokens."
            )

        padded_embeds = prompt_inputs_embeds.new_zeros(
            (
                prompt_inputs_embeds.shape[0],
                target_len,
                prompt_inputs_embeds.shape[2],
            )
        )
        padded_attention_mask = prompt_attention_mask.new_zeros(
            (prompt_attention_mask.shape[0], target_len)
        )
        for batch_idx, token_length_tensor in enumerate(prompt_token_lengths):
            token_length = int(token_length_tensor.item())
            if token_length == 0:
                continue
            token_embeds = prompt_inputs_embeds[batch_idx][prompt_token_mask[batch_idx]]
            if overlong_mask[batch_idx] and mask_overlong:
                token_embeds = token_embeds[-target_len:]
                token_length = target_len
            start = target_len - token_length
            padded_embeds[batch_idx, start:] = token_embeds
            padded_attention_mask[batch_idx, start:] = 1

        if not mask_overlong:
            overlong_mask = None
        return padded_embeds, padded_attention_mask, overlong_mask

    def _pad_response_forward_inputs(
        self,
        response_ids: torch.Tensor,
        target_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if target_len < 0:
            raise ValueError(
                "UniNaVid train metadata requires non-negative max_new_tokens."
            )
        current_len = int(response_ids.shape[1])
        if current_len > target_len:
            raise ValueError(
                "UniNaVid train metadata response length exceeds max_new_tokens."
            )

        response_mask = self._build_response_mask(response_ids)
        if current_len == target_len:
            return response_ids, response_mask

        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = 0
        pad_len = target_len - current_len
        id_pad = torch.full(
            (response_ids.shape[0], pad_len),
            int(pad_token_id),
            dtype=response_ids.dtype,
            device=response_ids.device,
        )
        mask_pad = torch.zeros(
            (response_ids.shape[0], pad_len),
            dtype=torch.bool,
            device=response_ids.device,
        )
        return (
            torch.cat([response_ids, id_pad], dim=1),
            torch.cat([response_mask, mask_pad], dim=1),
        )

    def _pad_response_inputs_with_logprobs(
        self,
        *,
        response_ids: torch.Tensor,
        response_mask: torch.Tensor,
        prev_logprobs: torch.Tensor,
        target_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        current_len = int(response_ids.shape[1])
        if current_len > target_len:
            raise ValueError(
                "UniNaVid train metadata response length exceeds max_new_tokens."
            )
        if current_len == target_len:
            return response_ids, response_mask, prev_logprobs

        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = 0
        pad_len = target_len - current_len
        id_pad = torch.full(
            (response_ids.shape[0], pad_len),
            int(pad_token_id),
            dtype=response_ids.dtype,
            device=response_ids.device,
        )
        mask_pad = torch.zeros(
            (response_ids.shape[0], pad_len),
            dtype=torch.bool,
            device=response_ids.device,
        )
        logprob_pad = prev_logprobs.new_zeros(
            (prev_logprobs.shape[0], pad_len, prev_logprobs.shape[2])
        )
        return (
            torch.cat([response_ids, id_pad], dim=1),
            torch.cat([response_mask.to(torch.bool), mask_pad], dim=1),
            torch.cat([prev_logprobs, logprob_pad], dim=1),
        )

    def _build_response_mask(self, response_ids: torch.Tensor) -> torch.Tensor:
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            return torch.ones_like(response_ids, dtype=torch.bool)
        return response_ids.ne(pad_token_id)

    def _compute_generation_score_logprobs(
        self,
        *,
        generated_scores: torch.Tensor,
        response_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        response_mask = self._build_response_mask(response_ids)
        prev_logprobs = self._gather_masked_token_logprobs(
            logits=generated_scores.float(),
            target=response_ids,
            mask=response_mask,
        )
        return prev_logprobs, response_mask

    def _gather_masked_token_logprobs(
        self,
        *,
        logits: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = mask.to(torch.bool)
        token_logprobs = logits.new_zeros((*target.shape, 1), dtype=torch.float32)
        if mask.any():
            active_logits = logits[mask].float()
            active_targets = target[mask]
            active_logprobs = compute_logprobs_from_logits(
                logits=active_logits.unsqueeze(1),
                target=active_targets.unsqueeze(1),
            ).squeeze(1)
            token_logprobs[mask] = active_logprobs.unsqueeze(-1)
        elif logits.requires_grad:
            # Alignment-only curriculum microbatches must still traverse the
            # LM graph so FSDP can finalize gradients accumulated under no_sync().
            token_logprobs = token_logprobs + logits.sum(dtype=torch.float32) * 0.0
        return token_logprobs

    def _embed_response_ids(self, response_ids: torch.Tensor) -> torch.Tensor:
        if hasattr(self.model, "get_input_embeddings"):
            return self.model.get_input_embeddings()(response_ids)
        if hasattr(self.model, "get_model") and hasattr(
            self.model.get_model(),
            "embed_tokens",
        ):
            return self.model.get_model().embed_tokens(response_ids)
        if hasattr(self.model, "model") and hasattr(self.model.model, "embed_tokens"):
            return self.model.model.embed_tokens(response_ids)
        raise AttributeError(
            "UniNaVid language model does not expose token embeddings."
        )

    def _compute_logits_from_embeds(
        self,
        *,
        prompt_inputs_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        response_ids: torch.Tensor,
        response_mask: torch.Tensor,
        return_hidden_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, int]:
        response_embeds = self._embed_response_ids(response_ids)
        full_inputs_embeds = torch.cat([prompt_inputs_embeds, response_embeds], dim=1)
        full_attention_mask = torch.cat(
            [prompt_attention_mask, response_mask.to(prompt_attention_mask.dtype)],
            dim=1,
        )
        outputs = self.model(
            inputs_embeds=full_inputs_embeds,
            attention_mask=full_attention_mask,
            use_cache=False,
            output_hidden_states=return_hidden_state,
            return_dict=True,
        )
        prompt_len = prompt_inputs_embeds.shape[1]
        response_logits = outputs.logits[:, prompt_len - 1 : -1, :]
        if not return_hidden_state:
            return response_logits
        if outputs.hidden_states is None:
            raise RuntimeError(
                "UniNaVid value head requires output_hidden_states=True."
            )
        final_hidden = outputs.hidden_states[-1]
        return response_logits, final_hidden, prompt_len

    def _compute_values_from_forward_inputs(
        self,
        *,
        prompt_inputs_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        response_ids: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> torch.Tensor:
        _, final_hidden, prompt_len = self._compute_logits_from_embeds(
            prompt_inputs_embeds=prompt_inputs_embeds,
            prompt_attention_mask=prompt_attention_mask,
            response_ids=response_ids,
            response_mask=response_mask,
            return_hidden_state=True,
        )
        return self._compute_value_from_pre_response_hidden(
            final_hidden=final_hidden,
            prompt_len=prompt_len,
        )

    def _compute_value_from_pre_response_hidden(
        self,
        *,
        final_hidden: torch.Tensor,
        prompt_len: int,
    ) -> torch.Tensor:
        if not hasattr(self, "value_head"):
            raise RuntimeError("UniNaVid value computation requires value_head.")
        value_feature = final_hidden[:, prompt_len - 1, :]
        if self._cfg_get(self.cfg, "detach_critic_input", default=False):
            value_feature = value_feature.detach()
        return self.value_head(value_feature).float()

    def _get_hidden_size_for_value_head(self) -> int:
        config = getattr(self.model, "config", None)
        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is None and hasattr(self.model, "get_model"):
            inner_config = getattr(self.model.get_model(), "config", None)
            hidden_size = getattr(inner_config, "hidden_size", None)
        if hidden_size is None:
            raise AttributeError("UniNaVid model config must expose hidden_size.")
        return int(hidden_size)

    def _move_images(self, images, *, device: torch.device):
        if images is None:
            return None
        if isinstance(images, torch.Tensor):
            if self.torch_dtype is None:
                return images.to(device=device)
            return images.to(device=device, dtype=self.torch_dtype)
        if isinstance(images, list):
            return [
                (
                    image.to(device=device)
                    if self.torch_dtype is None
                    else image.to(device=device, dtype=self.torch_dtype)
                )
                if isinstance(image, torch.Tensor)
                else image
                for image in images
            ]
        return images

    def _get_device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _initialize_fsdp_wrap_metadata(self) -> None:
        llm_backbone = getattr(self.model, "model", None)
        if llm_backbone is None:
            return

        self._set_fsdp_wrap_name(
            self._resolve_module(getattr(llm_backbone, "vision_tower", None)),
            "uninavid_vision_tower",
        )
        self._set_fsdp_wrap_name(
            self._resolve_module(getattr(llm_backbone, "mm_projector", None)),
            "uninavid_mm_projector",
        )
        self._set_fsdp_wrap_name(
            getattr(self.model, "lm_head", None),
            "uninavid_lm_head",
        )

    @staticmethod
    def _resolve_module(module: Any) -> nn.Module | None:
        if isinstance(module, nn.Module):
            return module
        if isinstance(module, (list, tuple)) and module:
            first_module = module[0]
            if isinstance(first_module, nn.Module):
                return first_module
        return None

    @staticmethod
    def _set_fsdp_wrap_name(module: nn.Module | None, wrap_name: str) -> None:
        if module is not None:
            module._fsdp_wrap_name = wrap_name

    def gradient_checkpointing_enable(self, **kwargs: Any) -> None:
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self) -> None:
        if hasattr(self.model, "gradient_checkpointing_disable"):
            self.model.gradient_checkpointing_disable()

    @staticmethod
    def _resolve_torch_dtype(torch_dtype: torch.dtype | None) -> torch.dtype:
        return torch_dtype or torch.float16

    @staticmethod
    def _apply_rope_scaling(config, model_max_length: int) -> None:
        orig_rope_scaling = getattr(config, "rope_scaling", None) or {"factor": 1}
        orig_rope_scaling_factor = orig_rope_scaling.get("factor", 1)
        orig_ctx_len = getattr(config, "max_position_embeddings", None)
        if orig_ctx_len:
            orig_ctx_len *= orig_rope_scaling_factor
            if model_max_length > orig_ctx_len:
                scaling_factor = float(math.ceil(model_max_length / orig_ctx_len))
                config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    @classmethod
    def _apply_multimodal_config_overrides(cls, config, cfg) -> None:
        vision_tower = cls._cfg_get(cfg, "vision_tower", default=None)
        if vision_tower is not None:
            config.mm_vision_tower = vision_tower

        image_processor = cls._cfg_get(cfg, "image_processor", default=None)
        if image_processor is not None:
            config.image_processor = image_processor

    @classmethod
    def _build_model_args(cls, cfg, model_name_or_path: str) -> SimpleNamespace:
        return SimpleNamespace(
            model_name_or_path=model_name_or_path,
            version=cls._cfg_get(cfg, "version", default="v0"),
            freeze_backbone=cls._cfg_get(cfg, "freeze_backbone", default=False),
            tune_mm_mlp_adapter=cls._cfg_get(
                cfg,
                "tune_mm_mlp_adapter",
                default=False,
            ),
            vision_tower=cls._cfg_get(cfg, "vision_tower", default=None),
            image_processor=cls._cfg_get(cfg, "image_processor", default=None),
            mm_vision_select_layer=cls._cfg_get(
                cfg,
                "mm_vision_select_layer",
                default=-1,
            ),
            pretrain_mm_mlp_adapter=cls._cfg_get(
                cfg,
                "pretrain_mm_mlp_adapter",
                default=None,
            ),
            mm_projector_type=cls._cfg_get(
                cfg,
                "mm_projector_type",
                default="linear",
            ),
            mm_use_im_start_end=cls._cfg_get(
                cfg,
                "mm_use_im_start_end",
                default=False,
            ),
            mm_use_im_patch_token=cls._cfg_get(
                cfg,
                "mm_use_im_patch_token",
                default=True,
            ),
            mm_vision_select_feature=cls._cfg_get(
                cfg,
                "mm_vision_select_feature",
                default="patch",
            ),
            compress_type=cls._cfg_get(cfg, "compress_type", default=None),
            run_type=cls._cfg_get(cfg, "run_type", default="train"),
        )

    @staticmethod
    def _smart_tokenizer_and_embedding_resize(
        *,
        special_tokens_dict: dict[str, str],
        tokenizer,
        model,
    ) -> None:
        num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer))

        if num_new_tokens > 0:
            input_embeddings = model.get_input_embeddings().weight.data
            output_embeddings = model.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                dim=0,
                keepdim=True,
            )
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                dim=0,
                keepdim=True,
            )

            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg

    @staticmethod
    def _apply_upstream_trainability(
        *,
        model: nn.Module,
        vision_tower: nn.Module,
        freeze_backbone: bool,
        tune_mm_mlp_adapter: bool,
        freeze_mm_mlp_adapter: bool,
        tune_vision_encoder: bool,
    ) -> None:
        if freeze_backbone:
            model.model.requires_grad_(False)

        if tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for param in model.get_model().mm_projector.parameters():
                param.requires_grad = True

        if freeze_mm_mlp_adapter:
            for param in model.get_model().mm_projector.parameters():
                param.requires_grad = False

        if tune_vision_encoder:
            vision_tower.requires_grad_(True)
        else:
            vision_tower.requires_grad_(False)

    @staticmethod
    def _cfg_get(cfg, *keys: str, default=None, required: bool = False):
        containers = [cfg]
        for attr in ("model", "data", "training", "algorithm"):
            nested = getattr(cfg, attr, None)
            if nested is not None:
                containers.append(nested)

        for key in keys:
            for container in containers:
                if isinstance(container, dict) and key in container:
                    return container[key]
                if hasattr(container, key):
                    return getattr(container, key)

        if required:
            joined_keys = ", ".join(keys)
            raise ValueError(f"Uni-NaVid config requires one of: {joined_keys}")
        return default
