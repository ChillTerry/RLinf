from subgoal_pipeline.artifacts import subgoals_to_info


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
