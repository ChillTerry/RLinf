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

import math
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType


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
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.torch_dtype = torch_dtype

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
        )

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
        if data is None:
            raise ValueError("Uni-NaVid sft_forward requires data.")

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
        if loss is None:
            raise ValueError("Uni-NaVid SFT forward expected outputs.loss.")
        return loss

    def _move_images(self, images, *, device: torch.device):
        if images is None:
            return None
        if isinstance(images, torch.Tensor):
            return self._move_image_tensor(images, device=device)
        if isinstance(images, list):
            return [
                self._move_image_tensor(image, device=device)
                if isinstance(image, torch.Tensor)
                else image
                for image in images
            ]
        return images

    def _move_image_tensor(
        self,
        image: torch.Tensor,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        if self.torch_dtype is None:
            return image.to(device=device)
        return image.to(device=device, dtype=self.torch_dtype)

    def _get_device(self) -> torch.device:
        return next(self.model.parameters()).device

    def default_forward(self, **kwargs):
        raise NotImplementedError("Uni-NaVid currently supports only SFT forward.")

    def predict_action_batch(self, **kwargs):
        raise NotImplementedError("Uni-NaVid action prediction is not wired yet.")
