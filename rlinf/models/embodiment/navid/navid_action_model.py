# Copyright 2025 The RLinf Authors.
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

import re
import warnings
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.modules.value_head import ValueHead
from rlinf.models.embodiment.navid.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IAMGE_SEPARATOR,
    IGNORE_INDEX,
    IMAGE_END_TOKEN,
    IMAGE_START_TOKEN,
    IMAGE_TOKEN_INDEX,
    NAVIGATION_IDENTIFIER,
    NAVIGATION_SPECIAL_TOKEN,
    VIDEO_END_SPECIAL_TOKEN,
    VIDEO_START_SPECIAL_TOKEN,
)
from rlinf.models.embodiment.navid.conversation import (
    SeparatorStyle,
    conv_templates,
)
from rlinf.models.embodiment.navid.mm_utils import (
    KeywordsStoppingCriteria,
    tokenizer_image_token,
)
from rlinf.utils.utils import compute_entropy_from_logits, compute_logprobs_from_logits


class NaVidForRLActionPrediction(nn.Module, BasePolicy):
    """
    A thin wrapper around NaVid (LLaVA-style) model to fit RLinf embodied policy interface.
    """

    _NAVID_FSDP_WRAP_NAMES = (
        "navid_vision_tower",
        "navid_mm_projector",
        "navid_lm_head",
    )

    def __init__(
        self,
        *,
        tokenizer,
        model,
        image_processor,
        action_dim: int,
        num_action_chunks: int,
        add_value_head: bool = False,
        hidden_size: int = 4096,
        max_prompt_length: int = 1024,
        max_history_len: Optional[int] = None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.action_dim = int(action_dim)
        self.num_action_chunks = int(num_action_chunks)
        self._history_rgb_tensor: dict[str, torch.Tensor | None] = {}
        self._max_history_len = (
            int(max_history_len) if max_history_len is not None else None
        )
        self.hidden_size = hidden_size
        self.max_prompt_length = max_prompt_length

        if add_value_head:
            self.value_head = ValueHead(
                input_dim=hidden_size,
                hidden_sizes=(512, 128),
                output_dim=1,
                activation="relu",
                bias_last=False,
            )

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
        return list(self._NAVID_FSDP_WRAP_NAMES)

    def _initialize_fsdp_wrap_metadata(self) -> None:
        llm_backbone = getattr(self.model, "model", None)
        if llm_backbone is None:
            return

        self._set_fsdp_wrap_name(
            self._resolve_module(getattr(llm_backbone, "vision_tower", None)),
            "navid_vision_tower",
        )
        self._set_fsdp_wrap_name(
            self._resolve_module(getattr(llm_backbone, "mm_projector", None)),
            "navid_mm_projector",
        )
        self._set_fsdp_wrap_name(getattr(self.model, "lm_head", None), "navid_lm_head")

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

    @classmethod
    def from_pretrained(
        cls,
        *,
        model_path: str,
        model_base: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        action_dim: int,
        num_action_chunks: int,
        add_value_head: bool = False,
        hidden_size: int = 4096,
        max_prompt_length: int = 1024,
        max_history_len: Optional[int] = None,
    ) -> "NaVidForRLActionPrediction":
        from rlinf.models.embodiment.navid.mm_utils import get_model_name_from_path
        from rlinf.models.embodiment.navid.model.builder import load_pretrained_model

        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path=model_path,
            model_base=model_base,
            model_name=model_name,
            load_8bit=False,
            load_4bit=False,
            device_map=None,
        )
        hidden_size = int(getattr(model.config, "hidden_size", hidden_size))

        policy = cls(
            tokenizer=tokenizer,
            model=model,
            image_processor=image_processor,
            action_dim=int(action_dim),
            num_action_chunks=int(num_action_chunks),
            add_value_head=add_value_head,
            hidden_size=hidden_size,
            max_prompt_length=max_prompt_length,
            max_history_len=max_history_len,
        )

        if torch_dtype is not None:
            policy.to(torch_dtype)

        return policy

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        raise NotImplementedError

    def default_forward(
        self,
        forward_inputs: Optional[dict[str, Any]] = None,
        compute_logprobs: bool = False,
        compute_entropy: bool = False,
        compute_values: bool = False,
        **kwargs,
    ):
        if forward_inputs is None:
            raise ValueError("NaVid default_forward requires forward_inputs.")

        input_ids = forward_inputs["input_ids"].long()
        attention_mask = forward_inputs["attention_mask"].long()
        pixel_values = forward_inputs["pixel_values"]
        response_token_ids = forward_inputs["response_token_ids"].long()
        response_mask = forward_inputs["response_mask"].long()

        device = self._get_device()
        input_ids = input_ids.to(device=device)
        attention_mask = attention_mask.to(device=device)
        pixel_values = pixel_values.to(device=device)
        response_token_ids = response_token_ids.to(device=device)
        response_mask = response_mask.to(device=device)

        full_input_ids, full_attention_mask, labels = self._build_teacher_forcing_batch(
            prompt_input_ids=input_ids,
            prompt_attention_mask=attention_mask,
            response_token_ids=response_token_ids,
            response_mask=response_mask,
        )

        outputs, aligned_labels = self._forward_teacher_forcing_multimodal(
            input_ids=full_input_ids,
            attention_mask=full_attention_mask,
            labels=labels,
            pixel_values=pixel_values,
            output_hidden_states=compute_values,
        )

        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = aligned_labels[..., 1:].contiguous()
        valid_mask = shift_labels.ne(IGNORE_INDEX)

        logprobs = None
        if compute_logprobs:
            token_logprobs = compute_logprobs_from_logits(
                logits=shift_logits,
                target=shift_labels,
            )
            logprobs = self._compact_valid_token_tensor(
                values=token_logprobs,
                valid_mask=valid_mask,
                target_length=response_token_ids.shape[1],
            )

        entropy = None
        if compute_entropy:
            token_entropy = compute_entropy_from_logits(shift_logits)
            entropy = self._compact_valid_token_tensor(
                values=token_entropy,
                valid_mask=valid_mask,
                target_length=response_token_ids.shape[1],
            )

        values = None
        if (
            compute_values
            and hasattr(self, "value_head")
            and outputs.hidden_states is not None
        ):
            shift_hidden_states = outputs.hidden_states[-1][..., :-1, :].contiguous()
            values = self._compute_response_level_values_from_shift_hidden_states(
                shift_hidden_states=shift_hidden_states,
                valid_mask=valid_mask,
            )

        return {
            "logprobs": logprobs,
            "entropy": entropy,
            "values": values,
        }

    def preprocess_env_obs(self, env_obs):
        out = dict(env_obs)
        images = out["main_images"]
        batch_images_np: list[np.ndarray] = []
        for i in range(int(images.shape[0])):
            img_np = images[i].detach().cpu().numpy()
            if img_np.dtype != np.uint8:
                img_np = np.clip(img_np, 0, 255).astype(np.uint8)
            batch_images_np.append(img_np)
        out["main_images"] = batch_images_np

        return out

    @torch.no_grad()
    def predict_action_batch(
        self,
        env_obs: dict[str, Any],
        **kwargs,
    ):
        env_obs = self.preprocess_env_obs(env_obs)

        device = self._get_device()
        gen_params = self._get_generation_params(**kwargs)

        task_descs = env_obs["task_descriptions"]  # [N_ENV]
        episode_ids = env_obs["states"].tolist()  # [N_ENV]
        images = env_obs["main_images"]  # [N_ENV, H, W, C]
        bsz = len(images)

        prompts, questions, convs = self._build_prompts_and_convs(
            bsz=bsz,
            task_descs=task_descs,
            conv_templates=conv_templates,
            conversation_template="vicuna_v1",
            default_image_token=DEFAULT_IMAGE_TOKEN,
            default_im_start_token=DEFAULT_IM_START_TOKEN,
            default_im_end_token=DEFAULT_IM_END_TOKEN,
        )

        special_tokens = self._build_special_token_tensors(
            image_start_token=IMAGE_START_TOKEN,
            image_end_token=IMAGE_END_TOKEN,
            video_start_token=VIDEO_START_SPECIAL_TOKEN,
            video_end_token=VIDEO_END_SPECIAL_TOKEN,
            navigation_token=NAVIGATION_SPECIAL_TOKEN,
            image_separator_token=IAMGE_SEPARATOR,
            device=device,
        )

        input_ids, attention_mask = self._tokenize_and_pad_prompts(
            prompts=prompts,
            tokenizer_image_token=tokenizer_image_token,
            image_token_index=IMAGE_TOKEN_INDEX,
            special_tokens=special_tokens,
            device=device,
        )

        stop_str, stopping_criteria = self._build_stopping_criteria(
            convs=convs, SeparatorStyle=SeparatorStyle, input_ids=input_ids
        )

        new_frames_tensor = self._preprocess_new_frames(images=images, device=device)
        images_for_model = self._accumulate_history_frames(
            new_frames_tensor=new_frames_tensor,
            episode_ids=episode_ids,
            device=device,
        )
        images_for_model = self._pad_history_frames_for_model(
            images_for_model=images_for_model
        )

        outputs = self._generate(
            questions=questions,
            input_ids=input_ids,
            attention_mask=attention_mask,
            images_for_model=images_for_model,
            stopping_criteria=stopping_criteria,
            gen_params=gen_params,
        )

        gen_texts = self._decode_generated_texts(
            outputs=outputs,
            input_token_len=input_ids.shape[1],
            stop_str=stop_str,
        )

        chunk_actions = self._parse_actions_from_texts(gen_texts=gen_texts)

        if hasattr(self, "value_head") and outputs.hidden_states is not None:
            prev_values = (
                self._compute_response_level_values_from_generation_hidden_states(
                    generation_hidden_states=outputs.hidden_states,
                )
            )
        else:
            prev_values = None

        input_token_len = input_ids.shape[1]
        generated_scores = torch.stack(tuple(outputs.scores), dim=1).float()
        generated_token_ids = outputs.sequences[
            :, input_token_len : input_token_len + generated_scores.shape[1]
        ]
        response_mask = self._build_response_mask(generated_token_ids)
        prev_logprobs = (
            compute_logprobs_from_logits(
                logits=generated_scores,
                target=generated_token_ids,
            ).float()
            * response_mask.float()
        )
        response_target_length = int(gen_params["max_new_tokens"])
        generated_token_ids, response_mask, prev_logprobs = (
            self._pad_generated_response_tensors(
                generated_token_ids=generated_token_ids,
                response_mask=response_mask,
                prev_logprobs=prev_logprobs,
                target_length=response_target_length,
            )
        )

        forward_inputs: dict[str, Any] = {}
        forward_inputs["input_ids"] = input_ids
        forward_inputs["attention_mask"] = attention_mask
        forward_inputs["pixel_values"] = torch.stack(images_for_model, dim=0)
        forward_inputs["response_token_ids"] = generated_token_ids
        forward_inputs["response_mask"] = response_mask
        forward_inputs["prompt_lengths"] = attention_mask.sum(dim=-1)
        forward_inputs["response_lengths"] = response_mask.sum(dim=-1)

        result = {
            "prev_logprobs": prev_logprobs,
            "prev_values": prev_values,
            "forward_inputs": forward_inputs,
        }
        return chunk_actions, result

    def _get_device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _compute_response_level_values_from_generation_hidden_states(
        self,
        *,
        generation_hidden_states: Any,
    ) -> torch.Tensor:
        token_hidden_states = generation_hidden_states[0]
        last_layer_hidden = token_hidden_states[-1]
        prompt_last_hidden = last_layer_hidden[:, -1, :]
        return self.value_head(prompt_last_hidden)

    def _compute_response_level_values_from_shift_hidden_states(
        self,
        *,
        shift_hidden_states: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if not valid_mask.any(dim=-1).all().item():
            raise ValueError(
                "NaVid PPO response-level value requires at least one valid response token per sample."
            )

        first_valid_indices = valid_mask.long().argmax(dim=-1)
        batch_indices = torch.arange(
            shift_hidden_states.shape[0], device=shift_hidden_states.device
        )
        prompt_hidden_states = shift_hidden_states[batch_indices, first_valid_indices]
        return self.value_head(prompt_hidden_states)

    def _build_response_mask(self, generated_token_ids: torch.Tensor) -> torch.Tensor:
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            return torch.ones_like(generated_token_ids, dtype=torch.long)

        response_mask = generated_token_ids.ne(int(pad_token_id)).long()
        zero_length_mask = response_mask.sum(dim=-1, keepdim=True).eq(0)
        if zero_length_mask.any():
            response_mask = torch.where(
                zero_length_mask,
                torch.ones_like(response_mask),
                response_mask,
            )
        return response_mask

    def _get_response_pad_token_id(self) -> int:
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token_id is not None:
            return int(pad_token_id)
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_token_id is not None:
            return int(eos_token_id)
        return 0

    def _pad_sequence_tensor(
        self,
        *,
        tensor: torch.Tensor,
        target_length: int,
        pad_value: int | float,
    ) -> torch.Tensor:
        current_length = int(tensor.shape[1])
        if current_length >= target_length:
            return tensor[:, :target_length]

        pad = tensor.new_full(
            (tensor.shape[0], target_length - current_length),
            fill_value=pad_value,
        )
        return torch.cat((tensor, pad), dim=1)

    def _pad_generated_response_tensors(
        self,
        *,
        generated_token_ids: torch.Tensor,
        response_mask: torch.Tensor,
        prev_logprobs: torch.Tensor,
        target_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pad_token_id = self._get_response_pad_token_id()
        padded_token_ids = self._pad_sequence_tensor(
            tensor=generated_token_ids,
            target_length=target_length,
            pad_value=pad_token_id,
        )
        padded_response_mask = self._pad_sequence_tensor(
            tensor=response_mask,
            target_length=target_length,
            pad_value=0,
        )
        padded_prev_logprobs = self._pad_sequence_tensor(
            tensor=prev_logprobs,
            target_length=target_length,
            pad_value=0.0,
        )
        return padded_token_ids, padded_response_mask, padded_prev_logprobs

    def _pad_history_frames_for_model(
        self, *, images_for_model: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        if self._max_history_len is None:
            return images_for_model

        padded_images: list[torch.Tensor] = []
        for history_frames in images_for_model:
            current_length = int(history_frames.shape[0])
            if current_length >= self._max_history_len:
                padded_images.append(history_frames[-self._max_history_len :])
                continue

            pad = history_frames.new_zeros(
                (self._max_history_len - current_length, *history_frames.shape[1:])
            )
            padded_images.append(torch.cat((pad, history_frames), dim=0))

        return padded_images

    def _build_teacher_forcing_batch(
        self,
        *,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        response_token_ids: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        full_input_ids = torch.cat((prompt_input_ids, response_token_ids), dim=-1)
        full_attention_mask = torch.cat((prompt_attention_mask, response_mask), dim=-1)

        response_labels = torch.where(
            response_mask.bool(),
            response_token_ids,
            torch.full_like(response_token_ids, IGNORE_INDEX),
        )
        prompt_labels = torch.full_like(prompt_input_ids, IGNORE_INDEX)
        labels = torch.cat((prompt_labels, response_labels), dim=-1)
        return full_input_ids, full_attention_mask, labels

    def _forward_teacher_forcing_multimodal(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        pixel_values: torch.Tensor,
        output_hidden_states: bool,
    ) -> tuple[Any, torch.Tensor]:
        prompts_for_update = [
            [NAVIGATION_IDENTIFIER] for _ in range(input_ids.shape[0])
        ]
        _, _, _, _, prepared_labels = self.model.prepare_inputs_labels_for_multimodal(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=None,
            labels=labels,
            images=pixel_values,
            prompts=prompts_for_update,
        )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            images=pixel_values,
            prompts=prompts_for_update,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        return outputs, prepared_labels

    def _compact_valid_token_tensor(
        self,
        *,
        values: torch.Tensor,
        valid_mask: torch.Tensor,
        target_length: int,
    ) -> torch.Tensor:
        compacted = values.new_zeros((values.shape[0], target_length))
        for batch_idx in range(values.shape[0]):
            valid_values = values[batch_idx][valid_mask[batch_idx]]
            valid_len = min(int(valid_values.shape[0]), target_length)
            if valid_len > 0:
                compacted[batch_idx, :valid_len] = valid_values[:valid_len]
        return compacted

    def _get_generation_params(self, **kwargs: Any) -> dict[str, Any]:
        do_sample = kwargs.get("do_sample", "True")
        temperature = float(kwargs.get("temperature", 0.2))
        max_new_tokens = int(kwargs.get("max_new_tokens", 1024) or 1024)
        return {
            "do_sample": bool(do_sample),
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
        }

    def _build_prompts_and_convs(
        self,
        *,
        bsz: int,
        task_descs: list[str],
        conv_templates,
        conversation_template: str,
        default_image_token: str,
        default_im_start_token: str,
        default_im_end_token: str,
    ) -> tuple[list[str], list[str], list]:
        conv_tmpl = conv_templates.get(conversation_template)
        if conv_tmpl is None:
            raise ValueError(
                f"Unknown NaVid conversation_template={conversation_template}. "
                f"Available: {sorted(conv_templates.keys())}"
            )

        navid_prompt_template = (
            "Imagine you are a robot programmed for navigation tasks. You have been given "
            "a video of historical observations and an image of the current observation <image>. "
            "Your assigned task is: '{}'. Analyze this series of images to decide your next move, "
            "which could involve turning left or right by a specific degree or moving forward "
            "a certain distance."
        )

        prompts: list[str] = []
        questions: list[str] = []
        convs: list = []

        for i in range(bsz):
            user_msg = navid_prompt_template.format(task_descs[i])
            question = user_msg.replace(default_image_token, "").replace("\n", "")

            if getattr(self.model.config, "mm_use_im_start_end", False):
                qs = (
                    default_im_start_token
                    + default_image_token
                    + default_im_end_token
                    + "\n"
                    + user_msg.replace("<image>", "")
                )
            else:
                qs = default_image_token + "\n" + user_msg.replace("<image>", "")

            conv = conv_tmpl.copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)

            questions.append(question)
            prompts.append(conv.get_prompt())
            convs.append(conv)

        return prompts, questions, convs

    @dataclass(frozen=True)
    class _SpecialTokens:
        image_start: torch.Tensor
        image_end: torch.Tensor
        video_start: torch.Tensor
        video_end: torch.Tensor
        navigation: torch.Tensor
        image_separator: torch.Tensor

    def _build_special_token_tensors(
        self,
        *,
        image_start_token: str,
        image_end_token: str,
        video_start_token: str,
        video_end_token: str,
        navigation_token: str,
        image_separator_token: str,
        device: torch.device,
    ) -> _SpecialTokens:
        def _tok(s: str) -> torch.Tensor:
            # drop leading BOS (match VLN-CE)
            return self.tokenizer(s, return_tensors="pt").input_ids[0][1:].to(device)

        return self._SpecialTokens(
            image_start=_tok(image_start_token),
            image_end=_tok(image_end_token),
            video_start=_tok(video_start_token),
            video_end=_tok(video_end_token),
            navigation=_tok(navigation_token),
            image_separator=_tok(image_separator_token),
        )

    def _tokenize_and_pad_prompts(
        self,
        *,
        prompts: list[str],
        tokenizer_image_token,
        image_token_index: int,
        special_tokens: _SpecialTokens,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_id_list: list[torch.Tensor] = []

        for prompt in prompts:
            token_prompt = tokenizer_image_token(
                prompt, self.tokenizer, image_token_index, return_tensors="pt"
            ).to(device)
            indices_to_replace = torch.where(token_prompt == image_token_index)[0]
            new_list: list[torch.Tensor] = []

            while indices_to_replace.numel() > 0:
                idx = indices_to_replace[0]
                new_list.append(token_prompt[:idx])
                new_list.append(special_tokens.video_start)
                new_list.append(special_tokens.image_separator)
                new_list.append(token_prompt[idx : idx + 1])  # keep IMAGE_TOKEN_INDEX
                new_list.append(special_tokens.video_end)
                new_list.append(special_tokens.image_start)
                new_list.append(special_tokens.image_end)
                new_list.append(special_tokens.navigation)
                token_prompt = token_prompt[idx + 1 :]
                indices_to_replace = torch.where(token_prompt == image_token_index)[0]

            if token_prompt.numel() > 0:
                new_list.append(token_prompt)

            input_ids_single = torch.cat(new_list, dim=0).unsqueeze(0)  # [1, seq]
            input_id_list.append(input_ids_single)

        bsz = len(input_id_list)
        max_len = self.max_prompt_length
        pad_id = int(self.tokenizer.pad_token_id or 0)

        input_ids = torch.full(
            (bsz, max_len), fill_value=pad_id, dtype=torch.long, device=device
        )
        attention_mask = torch.zeros((bsz, max_len), dtype=torch.long, device=device)

        for i, ids in enumerate(input_id_list):
            seq_len = min(ids.shape[1], max_len)
            input_ids[i, -seq_len:] = ids[0, :seq_len]
            attention_mask[i, -seq_len:] = 1

        return input_ids, attention_mask

    def _build_stopping_criteria(
        self, *, convs: list, SeparatorStyle, input_ids: torch.Tensor
    ) -> tuple[str, KeywordsStoppingCriteria]:
        stop_str = (
            convs[0].sep if convs[0].sep_style != SeparatorStyle.TWO else convs[0].sep2
        )
        stopping_criteria = KeywordsStoppingCriteria(
            [stop_str], self.tokenizer, input_ids
        )
        return stop_str, stopping_criteria

    def _preprocess_new_frames(
        self, *, images: list[np.ndarray], device: torch.device
    ) -> torch.Tensor:
        batch_image = np.asarray(images)
        new_frames_tensor = self.image_processor.preprocess(
            batch_image, return_tensors="pt"
        )["pixel_values"]  # [B, C, H, W]

        model_dtype = getattr(self.model, "dtype", None)
        if model_dtype in (torch.float16, torch.bfloat16):
            new_frames_tensor = new_frames_tensor.to(dtype=model_dtype)
        else:
            new_frames_tensor = new_frames_tensor.float()

        return new_frames_tensor.to(device=device)

    def _accumulate_history_frames(
        self,
        *,
        new_frames_tensor: torch.Tensor,
        episode_ids: list[int],
        device: torch.device,
    ) -> list[torch.Tensor]:
        bsz = int(new_frames_tensor.shape[0])
        images_for_model: list[torch.Tensor] = []

        if not self._history_rgb_tensor:
            self._history_episode_ids = episode_ids

        for i in range(bsz):
            ep_id = episode_ids[i]
            hist_ep_id = self._history_episode_ids[i]

            if ep_id != hist_ep_id:
                self._history_rgb_tensor.pop(hist_ep_id, None)
                self._history_episode_ids[i] = ep_id

            if ep_id not in self._history_rgb_tensor.keys():
                self._history_rgb_tensor[ep_id] = None

            new_frame = new_frames_tensor[i : i + 1]  # [1, C, H, W]
            if self._history_rgb_tensor[ep_id] is None:
                self._history_rgb_tensor[ep_id] = new_frame
            else:
                self._history_rgb_tensor[ep_id] = torch.cat(
                    (self._history_rgb_tensor[ep_id], new_frame), dim=0
                )

            if self._max_history_len is not None:
                hist = self._history_rgb_tensor[ep_id]
                if hist is not None and hist.shape[0] > self._max_history_len:
                    self._history_rgb_tensor[ep_id] = hist[-self._max_history_len :]

            images_for_model.append(self._history_rgb_tensor[ep_id].to(device=device))

        return images_for_model

    def _generate(
        self,
        *,
        questions: list[str],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        images_for_model: list[torch.Tensor],
        stopping_criteria,
        gen_params: dict[str, Any],
    ) -> Any:
        prompts_for_update = [[q] for q in questions]
        with torch.inference_mode():
            self.model.update_prompt(prompts_for_update)
            return self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=images_for_model,
                do_sample=gen_params["do_sample"],
                temperature=gen_params["temperature"],
                max_new_tokens=gen_params["max_new_tokens"],
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                return_dict_in_generate=True,
                output_scores=True,
                output_hidden_states=True,
            )

    def _decode_generated_texts(
        self,
        *,
        outputs: Any,
        input_token_len: int,
        stop_str: str,
    ) -> list[str]:
        gen_texts = self.tokenizer.batch_decode(
            outputs.sequences[:, input_token_len:], skip_special_tokens=True
        )
        return [
            text[: -len(stop_str)].strip() if text.endswith(stop_str) else text.strip()
            for text in gen_texts
        ]

    def _parse_actions_from_texts(self, *, gen_texts: list[str]) -> np.ndarray:
        chunk_actions = np.full(
            (len(gen_texts), self.num_action_chunks, 1), "no_op", dtype="<U12"
        )

        for i, gen_text in enumerate(gen_texts):
            text = gen_text.strip()
            match = re.search(r"-?\d+", text)
            num = int(match.group(0)) if match else 0

            actions: list[str] = []
            if "forward" in text:
                for _ in range(min(3, int(num / 25))):
                    actions.append("move_forward")
            elif "left" in text:
                for _ in range(min(3, int(num / 30))):
                    actions.append("turn_left")
            elif "right" in text:
                for _ in range(min(3, int(num / 30))):
                    actions.append("turn_right")
            elif "stop" in text:
                actions.append("stop")
            else:
                warnings.warn(
                    f"NaVid: action not found in generated text: {text!r}; padding with 'no_op'.",
                    stacklevel=2,
                )
                actions.append("no_op")

            num_actions = len(actions)
            fill_len = min(num_actions, self.num_action_chunks)
            chunk_actions[i, :fill_len, 0] = np.asarray(
                actions[:fill_len], dtype="<U12"
            )

        return chunk_actions
