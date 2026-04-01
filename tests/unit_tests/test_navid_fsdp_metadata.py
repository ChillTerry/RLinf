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

from types import SimpleNamespace

import torch.nn as nn

from rlinf.hybrid_engines.fsdp.utils import get_fsdp_wrap_policy
from rlinf.models.embodiment.navid.navid_action_model import NaVidForRLActionPrediction


class _DummyTokenizer:
    pad_token_id = 0


class _DummyImageProcessor:
    pass


class _DummyInnerNaVidModel(nn.Module):
    _no_split_modules = ["LlamaDecoderLayer"]

    def __init__(self):
        super().__init__()
        self.decoder = LlamaDecoderLayer()
        self.vision_tower = nn.Linear(2, 2, bias=False)
        self.mm_projector = nn.Sequential(nn.Linear(2, 2), nn.GELU())


class LlamaDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 2)


class _DummyFSDPConfig(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)


class _DummyWrappedNaVidModel(nn.Module):
    _no_split_modules = ["LlamaDecoderLayer"]

    def __init__(self):
        super().__init__()
        self.model = _DummyInnerNaVidModel()
        self.lm_head = nn.Linear(2, 3, bias=False)
        self.config = SimpleNamespace(hidden_size=2)

    def update_prompt(self, prompts=None):
        self.prompts = prompts


def test_navid_wrapper_exposes_fsdp_wrap_metadata():
    policy = NaVidForRLActionPrediction(
        tokenizer=_DummyTokenizer(),
        model=_DummyWrappedNaVidModel(),
        image_processor=_DummyImageProcessor(),
        action_dim=1,
        num_action_chunks=3,
        hidden_size=2,
    )

    assert policy._no_split_modules == ["LlamaDecoderLayer"]
    assert policy._no_split_names == [
        "navid_vision_tower",
        "navid_mm_projector",
        "navid_lm_head",
    ]
    assert policy.model.model.vision_tower._fsdp_wrap_name == "navid_vision_tower"
    assert policy.model.model.mm_projector._fsdp_wrap_name == "navid_mm_projector"
    assert policy.model.lm_head._fsdp_wrap_name == "navid_lm_head"


def test_navid_wrapper_handles_list_backed_vision_tower_metadata():
    model = _DummyWrappedNaVidModel()
    list_backed_vision_tower = model.model.vision_tower
    model.model.__dict__["vision_tower"] = [list_backed_vision_tower]

    policy = NaVidForRLActionPrediction(
        tokenizer=_DummyTokenizer(),
        model=model,
        image_processor=_DummyImageProcessor(),
        action_dim=1,
        num_action_chunks=3,
        hidden_size=2,
    )

    assert policy._no_split_names[0] == "navid_vision_tower"
    assert list_backed_vision_tower._fsdp_wrap_name == "navid_vision_tower"


def test_navid_wrapper_builds_fsdp_wrap_policy():
    policy = NaVidForRLActionPrediction(
        tokenizer=_DummyTokenizer(),
        model=_DummyWrappedNaVidModel(),
        image_processor=_DummyImageProcessor(),
        action_dim=1,
        num_action_chunks=3,
        hidden_size=2,
    )

    fsdp_config = _DummyFSDPConfig(use_orig_params=False, wrap_policy={})
    wrap_policy = get_fsdp_wrap_policy(
        policy,
        config=fsdp_config,
        is_lora=False,
        model_type="navid",
    )

    assert wrap_policy is not None
