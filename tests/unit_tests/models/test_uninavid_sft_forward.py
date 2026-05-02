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

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.uninavid.uninavid_action_model import (
    UniNaVidForActionPrediction,
)


class FakeUniNaVidModel(nn.Module):
    def __init__(self, output=None):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))
        self.loss = torch.tensor(7.0)
        self.output = output
        self.call_kwargs = None

    def forward(self, **kwargs):
        self.call_kwargs = kwargs
        if self.output is not None:
            return self.output
        return SimpleNamespace(loss=self.loss)


class FakeTokenizer:
    def __init__(self, initial_length):
        self.length = initial_length
        self.pad_token = None

    def __len__(self):
        return self.length

    def add_special_tokens(self, special_tokens_dict):
        self.pad_token = special_tokens_dict["pad_token"]
        self.length += 1
        return 1


class FakeEmbeddingResizeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_embeddings = nn.Embedding(3, 2)
        self.output_embeddings = nn.Embedding(3, 2)
        with torch.no_grad():
            self.input_embeddings.weight.copy_(
                torch.tensor([[1.0, 3.0], [5.0, 7.0], [9.0, 11.0]])
            )
            self.output_embeddings.weight.copy_(
                torch.tensor([[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]])
            )

    def resize_token_embeddings(self, new_size):
        old_input = self.input_embeddings.weight.detach().clone()
        old_output = self.output_embeddings.weight.detach().clone()
        self.input_embeddings = nn.Embedding(new_size, old_input.shape[1])
        self.output_embeddings = nn.Embedding(new_size, old_output.shape[1])
        with torch.no_grad():
            self.input_embeddings.weight[: old_input.shape[0]].copy_(old_input)
            self.output_embeddings.weight[: old_output.shape[0]].copy_(old_output)

    def get_input_embeddings(self):
        return self.input_embeddings

    def get_output_embeddings(self):
        return self.output_embeddings


class FakeVisionTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.image_processor = object()
        self.to_kwargs = None

    def to(self, *args, **kwargs):
        self.to_kwargs = kwargs
        return super().to(*args, **kwargs)


class FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(2, 2)
        self.mm_projector = nn.Linear(2, 2)


class FakeTrainabilityModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = FakeBackbone()
        self.head = nn.Linear(2, 2)

    def get_model(self):
        return self.model


class FakeFromPretrainedBackbone(FakeBackbone):
    def __init__(self):
        super().__init__()
        self.initialize_vision_modules_kwargs = None

    def initialize_vision_modules(self, **kwargs):
        self.initialize_vision_modules_kwargs = kwargs


class FakeFromPretrainedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace()
        self.model = FakeFromPretrainedBackbone()
        self.head = nn.Linear(2, 2)
        self.vision_tower = FakeVisionTower()
        self.initialize_vision_tokenizer_kwargs = None

    def get_model(self):
        return self.model

    def get_vision_tower(self):
        return self.vision_tower

    def initialize_vision_tokenizer(self, *args, **kwargs):
        self.initialize_vision_tokenizer_kwargs = kwargs


def _make_policy():
    return UniNaVidForActionPrediction(
        tokenizer=object(),
        model=FakeUniNaVidModel(),
        image_processor=object(),
        torch_dtype=torch.bfloat16,
    )


def _make_batch(images):
    return {
        "input_ids": torch.ones(2, 3, dtype=torch.long),
        "attention_mask": torch.ones(2, 3, dtype=torch.long),
        "labels": torch.zeros(2, 3, dtype=torch.long),
        "images": images,
        "prompts": [["prompt one"], ["prompt two"]],
    }


def test_sft_forward_returns_inner_loss_and_moves_tensor_images():
    policy = _make_policy()
    batch = _make_batch(torch.ones(2, 3, 4, 4, dtype=torch.float32))

    loss = policy(forward_type=ForwardType.SFT, data=batch)

    assert loss is policy.model.loss
    call_kwargs = policy.model.call_kwargs
    device = next(policy.model.parameters()).device
    assert call_kwargs["input_ids"].device == device
    assert call_kwargs["attention_mask"].device == device
    assert call_kwargs["labels"].device == device
    assert call_kwargs["images"].device == device
    assert call_kwargs["images"].dtype == policy.torch_dtype
    assert call_kwargs["prompts"] == batch["prompts"]
    assert call_kwargs["use_cache"] is False
    assert call_kwargs["return_dict"] is True


def test_sft_forward_moves_and_casts_list_images():
    policy = _make_policy()
    batch = _make_batch(
        [
            torch.ones(3, 4, 4, dtype=torch.float32),
            torch.zeros(3, 4, 4, dtype=torch.float32),
        ]
    )

    loss = policy(forward_type=ForwardType.SFT, data=batch)

    assert loss is policy.model.loss
    call_kwargs = policy.model.call_kwargs
    device = next(policy.model.parameters()).device
    assert isinstance(call_kwargs["images"], list)
    assert len(call_kwargs["images"]) == len(batch["images"])
    for image in call_kwargs["images"]:
        assert image.device == device
        assert image.dtype == policy.torch_dtype
    assert call_kwargs["prompts"] == batch["prompts"]
    assert call_kwargs["use_cache"] is False
    assert call_kwargs["return_dict"] is True


def test_sft_forward_accepts_batch_fields_as_kwargs():
    policy = _make_policy()
    batch = _make_batch(torch.ones(2, 3, 4, 4, dtype=torch.float32))

    loss = policy(forward_type=ForwardType.SFT, **batch)

    assert loss is policy.model.loss
    assert policy.model.call_kwargs["prompts"] == batch["prompts"]


def test_sft_forward_raises_value_error_when_output_has_no_loss():
    policy = UniNaVidForActionPrediction(
        tokenizer=object(),
        model=FakeUniNaVidModel(output=SimpleNamespace()),
        image_processor=object(),
        torch_dtype=torch.bfloat16,
    )
    batch = _make_batch(torch.ones(2, 3, 4, 4, dtype=torch.float32))

    with pytest.raises(ValueError):
        policy(forward_type=ForwardType.SFT, data=batch)


def test_resolve_torch_dtype_defaults_to_float16():
    assert UniNaVidForActionPrediction._resolve_torch_dtype(None) is torch.float16
    assert (
        UniNaVidForActionPrediction._resolve_torch_dtype(torch.bfloat16)
        is torch.bfloat16
    )


def test_from_pretrained_stores_and_loads_with_resolved_default_dtype(monkeypatch):
    import transformers

    from rlinf.models.embodiment.uninavid import model as uninavid_model

    load_kwargs = {}
    fake_model = FakeFromPretrainedModel()
    fake_tokenizer = SimpleNamespace(pad_token=None, unk_token="<unk>")
    fake_cfg = SimpleNamespace(
        model_path="fake-vicuna",
        version="v0.5",
        vision_tower="fake-eva",
        device="cpu",
    )

    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(max_position_embeddings=2048),
    )
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: fake_tokenizer,
    )

    def fake_model_from_pretrained(*args, **kwargs):
        load_kwargs.update(kwargs)
        return fake_model

    monkeypatch.setattr(
        uninavid_model.LlavaLlamaAttForCausalLM,
        "from_pretrained",
        fake_model_from_pretrained,
    )

    policy = UniNaVidForActionPrediction.from_pretrained(
        cfg=fake_cfg,
        torch_dtype=None,
    )

    assert policy.torch_dtype is torch.float16
    assert load_kwargs["torch_dtype"] is torch.float16
    assert fake_model.vision_tower.to_kwargs["dtype"] is torch.float16


def test_smart_tokenizer_resize_initializes_new_embeddings_to_old_average():
    tokenizer = FakeTokenizer(initial_length=3)
    model = FakeEmbeddingResizeModel()

    UniNaVidForActionPrediction._smart_tokenizer_and_embedding_resize(
        special_tokens_dict={"pad_token": "[PAD]"},
        tokenizer=tokenizer,
        model=model,
    )

    assert len(tokenizer) == 4
    assert tokenizer.pad_token == "[PAD]"
    torch.testing.assert_close(
        model.get_input_embeddings().weight[-1],
        torch.tensor([5.0, 7.0]),
    )
    torch.testing.assert_close(
        model.get_output_embeddings().weight[-1],
        torch.tensor([6.0, 8.0]),
    )


def test_apply_upstream_trainability_matches_stage_one_overrides():
    model = FakeTrainabilityModel()
    vision_tower = FakeVisionTower()

    UniNaVidForActionPrediction._apply_upstream_trainability(
        model=model,
        vision_tower=vision_tower,
        freeze_backbone=True,
        tune_mm_mlp_adapter=True,
        freeze_mm_mlp_adapter=True,
        tune_vision_encoder=True,
    )

    assert not any(p.requires_grad for p in model.model.backbone.parameters())
    assert not any(p.requires_grad for p in model.head.parameters())
    assert not any(p.requires_grad for p in model.get_model().mm_projector.parameters())
    assert all(p.requires_grad for p in vision_tower.parameters())


def test_apply_upstream_trainability_keeps_stage_one_defaults():
    model = FakeTrainabilityModel()
    vision_tower = FakeVisionTower()

    UniNaVidForActionPrediction._apply_upstream_trainability(
        model=model,
        vision_tower=vision_tower,
        freeze_backbone=False,
        tune_mm_mlp_adapter=False,
        freeze_mm_mlp_adapter=False,
        tune_vision_encoder=False,
    )

    assert all(p.requires_grad for p in model.parameters())
    assert not any(p.requires_grad for p in vision_tower.parameters())
