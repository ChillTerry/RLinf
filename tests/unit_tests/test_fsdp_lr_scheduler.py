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

import torch

from rlinf.hybrid_engines.fsdp.utils import get_lr_scheduler


def test_cosine_scheduler_supports_transformers_431_without_min_lr():
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.Adam([parameter], lr=1.0)

    scheduler = get_lr_scheduler(
        lr_scheduler="cosine",
        optimizer=optimizer,
        num_warmup_steps=1,
        num_training_steps=4,
    )

    assert scheduler.__class__.__name__ == "LambdaLR"
