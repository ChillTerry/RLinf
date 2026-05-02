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
from omegaconf import OmegaConf

from rlinf.models.embodiment.uninavid.constants import IGNORE_INDEX
from rlinf.models.embodiment.uninavid.train.data import (
    DataCollatorForSupervisedDataset,
    _load_decord_for_raw_video,
    _resolve_data_path,
)


def test_uninavid_data_collator_pads_truncates_stacks_images_and_keeps_prompts():
    tokenizer = SimpleNamespace(pad_token_id=0, model_max_length=4)
    collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    image_shape = (3, 2, 2)
    instances = [
        {
            "input_ids": torch.tensor([11, 12, 13, 14, 15]),
            "labels": torch.tensor([21, 22, 23, 24, 25]),
            "image": torch.ones(image_shape),
            "prompt": ["turn left"],
        },
        {
            "input_ids": torch.tensor([31, 32]),
            "labels": torch.tensor([41, 42]),
            "image": torch.zeros(image_shape),
            "prompt": ["turn right"],
        },
    ]

    batch = collator(instances)

    assert torch.equal(
        batch["input_ids"],
        torch.tensor(
            [
                [11, 12, 13, 14],
                [31, 32, tokenizer.pad_token_id, tokenizer.pad_token_id],
            ]
        ),
    )
    assert torch.equal(
        batch["labels"],
        torch.tensor(
            [
                [21, 22, 23, 24],
                [41, 42, IGNORE_INDEX, IGNORE_INDEX],
            ]
        ),
    )
    assert torch.equal(
        batch["attention_mask"], batch["input_ids"].ne(tokenizer.pad_token_id)
    )
    assert torch.equal(
        batch["images"],
        torch.stack([torch.ones(image_shape), torch.zeros(image_shape)]),
    )
    assert batch["prompts"] == [["turn left"], ["turn right"]]


def test_uninavid_data_collator_keeps_differently_shaped_images_as_list():
    tokenizer = SimpleNamespace(pad_token_id=0, model_max_length=4)
    collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    first_image = torch.ones(3, 2, 2)
    second_image = torch.zeros(3, 4, 4)
    instances = [
        {
            "input_ids": torch.tensor([11, 12]),
            "labels": torch.tensor([21, 22]),
            "image": first_image,
        },
        {
            "input_ids": torch.tensor([31, 32]),
            "labels": torch.tensor([41, 42]),
            "image": second_image,
        },
    ]

    batch = collator(instances)

    assert batch["images"] == [first_image, second_image]


def test_resolve_data_path_uses_first_item_from_omegaconf_list():
    data_paths = OmegaConf.create(["/tmp/a.json"])

    assert _resolve_data_path(data_paths) == "/tmp/a.json"


def test_missing_decord_error_mentions_raw_video_loading(monkeypatch):
    def fail_import(name):
        if name == "decord":
            raise ModuleNotFoundError("No module named 'decord'")
        raise AssertionError(name)

    monkeypatch.setattr(
        "rlinf.models.embodiment.uninavid.train.data.importlib.import_module",
        fail_import,
    )

    with pytest.raises(ModuleNotFoundError, match="decord.*raw video loading"):
        _load_decord_for_raw_video()
