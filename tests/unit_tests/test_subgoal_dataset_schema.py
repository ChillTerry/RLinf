from types import SimpleNamespace

import numpy as np

from subgoal_pipeline.artifacts import (
    drop_trailing_subgoals_near_final_goal,
    subgoals_to_info,
)


def test_subgoals_are_stored_in_info_without_final_goal():
    original_episode = {
        "episode_id": "7",
        "goals": [{"position": [9.0, 0.0, 1.0], "radius": 3.0}],
        "info": {"geodesic_distance": 12.0},
    }
    generated = [
        {
            "subgoal_id": 0,
            "subgoal_position": [1.0, 0.0, 2.0],
            "is_final_goal": False,
        },
        {
            "subgoal_id": 1,
            "subgoal_position": [9.0, 0.0, 1.0],
            "is_final_goal": True,
        },
    ]

    info = subgoals_to_info(generated, original_episode)

    assert info == {
        "geodesic_distance": 12.0,
        "subgoals": [{"position": [1.0, 0.0, 2.0]}],
    }
    assert original_episode["goals"] == [
        {"position": [9.0, 0.0, 1.0], "radius": 3.0}
    ]
    assert "subgoals" not in original_episode["info"]


def test_final_goal_only_payload_writes_empty_subgoal_list():
    info = subgoals_to_info(
        [
            {
                "subgoal_id": 0,
                "subgoal_position": [9.0, 0.0, 1.0],
                "is_final_goal": True,
            }
        ],
        {"episode_id": "7", "goals": [{"position": [9.0, 0.0, 1.0]}]},
    )

    assert info == {"subgoals": []}


def test_trailing_subgoals_inside_final_goal_region_are_removed():
    sim = SimpleNamespace(
        geodesic_distance=lambda source, target: float(
            np.linalg.norm(np.asarray(source) - np.asarray(target))
        )
    )
    generated = [
        {
            "subgoal_id": 0,
            "subgoal_position": [4.0, 0.0, 0.0],
            "is_final_goal": False,
        },
        {
            "subgoal_id": 1,
            "subgoal_position": [8.0, 0.0, 0.0],
            "is_final_goal": False,
        },
        {
            "subgoal_id": 2,
            "subgoal_position": [10.0, 0.0, 0.0],
            "is_final_goal": True,
        },
    ]

    filtered = drop_trailing_subgoals_near_final_goal(
        generated,
        sim,
        final_goal_position=[10.0, 0.0, 0.0],
        exclusion_distance=3.0,
    )

    assert [item["subgoal_position"] for item in filtered] == [
        [4.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
    ]
    assert [item["subgoal_id"] for item in filtered] == [0, 1]
    assert len(generated) == 3
