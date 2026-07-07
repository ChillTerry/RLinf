import math

import numpy as np
import pytest

from subgoal_pipeline import build_dataset_geodesic
from subgoal_pipeline.replay import MemoryStep


class FakeSim:
    """Stand-in for habitat sim; geodesic_distance is scriptable."""

    def __init__(self, geodesic_fn=None):
        self._fn = geodesic_fn or (lambda a, b: math.dist(list(a), list(b)))

    def geodesic_distance(self, a, b):
        return self._fn(a, b)


def _step(i, pos):
    return MemoryStep(
        step=int(i),
        rgb=np.zeros(1, dtype=np.uint8),
        agent_position=[float(v) for v in pos],
        agent_rotation=[0.0, 0.0, 0.0, 1.0],
    )


def test_evenly_spaced_with_single_tail_removal():
    # 11 steps on a line at x=0..10; D=3; final goal at x=10 (stop frame).
    steps = [_step(i, [float(i), 0.0, 0.0]) for i in range(11)]
    selected = build_dataset_geodesic.select_geodesic_subgoal_steps(
        steps, FakeSim(), subgoal_distance=3.0, final_goal_position=[10.0, 0.0, 0.0]
    )
    # Greedy picks 3, 6, 9; tail 9->10 == 1 < 3 dropped; 6->10 == 4 kept.
    assert selected == [3, 6]


def test_iterative_double_tail_removal():
    # start(-3,0) -> A(0,0) -> B(3,0) -> final(1.5, 2).
    # Greedy picks A (index 3) and B (index 6).
    # B->final == 2.5 < 3 dropped; A->final == 2.5 < 3 dropped.
    positions = [
        [-3.0, 0.0, 0.0],
        [-2.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [1.5, 2.0, 0.0],
    ]
    steps = [_step(i, p) for i, p in enumerate(positions)]
    selected = build_dataset_geodesic.select_geodesic_subgoal_steps(
        steps, FakeSim(), subgoal_distance=3.0, final_goal_position=[1.5, 2.0, 0.0]
    )
    assert selected == []


def test_path_shorter_than_distance_yields_no_intermediates():
    steps = [_step(0, [0.0, 0.0, 0.0]), _step(1, [0.5, 0.0, 0.0]), _step(2, [1.0, 0.0, 0.0])]
    selected = build_dataset_geodesic.select_geodesic_subgoal_steps(
        steps, FakeSim(), subgoal_distance=3.0, final_goal_position=[1.0, 0.0, 0.0]
    )
    assert selected == []


def test_too_few_steps_returns_empty():
    steps = [_step(0, [0.0, 0.0, 0.0])]
    selected = build_dataset_geodesic.select_geodesic_subgoal_steps(
        steps, FakeSim(), subgoal_distance=3.0, final_goal_position=[0.0, 0.0, 0.0]
    )
    assert selected == []


def test_inf_geodesic_falls_back_to_euclidean(capsys):
    # Force the (start -> step at x=3) call to return inf; euclidean=3 should still select it.
    def fn(a, b):
        a = list(a)
        b = list(b)
        if math.dist(a, b) >= 3.0 and abs(a[0] - b[0]) >= 3.0 and a[1] == 0.0 and b[1] == 0.0:
            return float("inf")
        return math.dist(a, b)

    steps = [_step(i, [float(i), 0.0, 0.0]) for i in range(5)]
    selected = build_dataset_geodesic.select_geodesic_subgoal_steps(
        steps, FakeSim(fn), subgoal_distance=3.0, final_goal_position=[4.0, 0.0, 0.0]
    )
    # Greedy picks 3 (via euclidean fallback); tail 3->4 == 1 < 3 dropped -> [].
    assert selected == []
    out = capsys.readouterr().out
    assert "geodesic_distance not finite" in out


def test_selector_never_returns_stop_frame_as_intermediate():
    # D=1 selects every step; stop frame (index 5) is the last step.
    # final_goal at x=10 is far from the stop frame (x=5), so tail-removal does NOT pop it.
    # Without the exclusion the stop frame would be returned as an intermediate and collide
    # with the final sub-goal's best_step in _build_geodesic_subgoal_dicts.
    steps = [_step(i, [float(i), 0.0, 0.0]) for i in range(6)]
    selected = build_dataset_geodesic.select_geodesic_subgoal_steps(
        steps, FakeSim(), subgoal_distance=1.0, final_goal_position=[10.0, 0.0, 0.0]
    )
    assert 5 not in selected
    assert all(i < len(steps) - 1 for i in selected)


def test_subgoal_distance_is_required():
    parser = build_dataset_geodesic.build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_gpt_only_args_are_absent():
    parser = build_dataset_geodesic.build_arg_parser()
    dests = {action.dest for action in parser._actions}
    for gpt_arg in [
        "model", "reasoning_effort", "max_output_tokens", "max_frames",
        "frame_stride", "min_step_gap", "image_format", "jpeg_quality",
        "base_url", "api_key", "user_agent", "usage_log_name",
    ]:
        assert gpt_arg not in dests, f"GPT-only arg --{gpt_arg} should not exist"


def test_subgoal_distance_parsed():
    parser = build_dataset_geodesic.build_arg_parser()
    args = parser.parse_args(["--subgoal_distance", "3.5"])
    assert args.subgoal_distance == 3.5
