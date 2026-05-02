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

import inspect

from rlinf.workers.sft.fsdp_vla_sft_worker import FSDPVlaSftWorker


def test_build_dataloader_has_uninavid_sft_branch():
    source = inspect.getsource(FSDPVlaSftWorker.build_dataloader)

    assert "SupportedModel.UNINAVID" in source
    assert "build_uninavid_sft_dataloader" in source


def test_get_train_model_output_uses_uninavid_sft_forward_path():
    source = inspect.getsource(FSDPVlaSftWorker.get_train_model_output)

    assert "SupportedModel.UNINAVID" in source
    assert "ForwardType.SFT" in source
