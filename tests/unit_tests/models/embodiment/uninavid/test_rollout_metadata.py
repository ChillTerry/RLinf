import pytest
import torch

from rlinf.models.embodiment.uninavid.nav_rollout import select_slot_rgb_frames


def test_select_slot_rgb_frames_uses_rgb_frame_history_lengths():
    env_obs = {
        "rgb_frame_history": torch.tensor(
            [
                [[[[1]]], [[[2]]], [[[3]]]],
                [[[[4]]], [[[5]]], [[[6]]]],
            ],
            dtype=torch.uint8,
        ),
        "rgb_frame_history_lengths": torch.tensor([2, 1], dtype=torch.long),
    }

    slot0 = select_slot_rgb_frames(env_obs, 0)
    slot1 = select_slot_rgb_frames(env_obs, 1)

    assert [frame.tolist() for frame in slot0] == [[[[1]]], [[[2]]]]
    assert [frame.tolist() for frame in slot1] == [[[[4]]]]


def test_select_slot_rgb_frames_requires_rgb_frame_history():
    with pytest.raises(KeyError, match="rgb_frame_history"):
        select_slot_rgb_frames({"wrist_images": torch.zeros((1, 1, 1, 1))}, 0)


def test_select_slot_rgb_frames_requires_rgb_frame_history_lengths():
    with pytest.raises(KeyError, match="rgb_frame_history_lengths"):
        select_slot_rgb_frames(
            {"rgb_frame_history": torch.zeros((1, 1, 1, 1, 1), dtype=torch.uint8)},
            0,
        )
