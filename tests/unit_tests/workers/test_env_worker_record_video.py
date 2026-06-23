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

from rlinf.workers.env.env_worker import EnvWorker


def test_env_worker_skips_record_video_wrapper_for_habitat():
    worker = object.__new__(EnvWorker)
    env_cfg = SimpleNamespace(
        env_type="habitat",
        video_cfg=SimpleNamespace(save_video=True),
    )

    assert worker._should_wrap_record_video(env_cfg) is False


def test_env_worker_keeps_record_video_wrapper_for_non_habitat():
    worker = object.__new__(EnvWorker)
    env_cfg = SimpleNamespace(
        env_type="maniskill",
        video_cfg=SimpleNamespace(save_video=True),
    )

    assert worker._should_wrap_record_video(env_cfg) is True


def test_env_worker_does_not_wrap_record_video_when_disabled():
    worker = object.__new__(EnvWorker)
    env_cfg = SimpleNamespace(
        env_type="habitat",
        video_cfg=SimpleNamespace(save_video=False),
    )

    assert worker._should_wrap_record_video(env_cfg) is False
