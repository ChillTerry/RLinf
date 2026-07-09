import pytest

from subgoal_pipeline import build_dataset_geodesic as bdg


def test_parse_gpus_basic():
    assert bdg.parse_gpus("0,1,2,3") == [0, 1, 2, 3]
    assert bdg.parse_gpus("0") == [0]
    assert bdg.parse_gpus(" 1 , 2 ") == [1, 2]


def test_parse_gpus_rejects_bad():
    for bad in ["", " ", "0,a", "0,-1", "0,1.5"]:
        with pytest.raises((ValueError, RuntimeError)):
            bdg.parse_gpus(bad)


def test_validate_num_processes_vs_gpus():
    bdg.validate_num_processes_vs_gpus(4, 4)  # ok
    bdg.validate_num_processes_vs_gpus(8, 4)  # ok
    with pytest.raises((ValueError, RuntimeError)):
        bdg.validate_num_processes_vs_gpus(2, 4)


def test_bucket_groups_by_scene():
    # group = (trajectory_id, episode_ids, representative_id)
    groups = [(10, [1, 2], 1), (20, [3], 3), (30, [5, 6], 5)]
    source = {
        1: {"scene_id": "sceneA"},
        3: {"scene_id": "sceneB"},
        5: {"scene_id": "sceneA"},
    }
    buckets = bdg.bucket_groups_by_scene(groups, source)
    assert set(buckets.keys()) == {"sceneA", "sceneB"}
    assert [g[0] for g in buckets["sceneA"]] == [10, 30]
    assert [g[0] for g in buckets["sceneB"]] == [20]


def test_content_scene_key_uses_habitat_scene_basename():
    assert bdg.content_scene_key("mp3d/uNb9QFRL6hY/uNb9QFRL6hY.glb") == "uNb9QFRL6hY"
    assert bdg.content_scene_key("VLN-CE/scene_dataset/mp3d/r1Q1Z4BcV1o/r1Q1Z4BcV1o.glb") == "r1Q1Z4BcV1o"
    assert bdg.content_scene_key("plain_scene") == "plain_scene"


def test_assign_scenes_to_gpus_round_robin():
    scenes = ["s0", "s1", "s2", "s3", "s4"]
    gpus = [0, 1, 2]
    assignment = bdg.assign_scenes_to_gpus(scenes, gpus)
    assert assignment[0] == ["s0", "s3"]
    assert assignment[1] == ["s1", "s4"]
    assert assignment[2] == ["s2"]


def test_compute_workers_per_gpu_even():
    gpus = [0, 1, 2, 3]
    scenes_per_gpu = {0: 5, 1: 5, 2: 5, 3: 5}
    out = bdg.compute_workers_per_gpu(8, gpus, scenes_per_gpu)
    assert out == {0: 2, 1: 2, 2: 2, 3: 2}


def test_compute_workers_per_gpu_remainder():
    gpus = [0, 1, 2, 3]
    scenes_per_gpu = {0: 5, 1: 5, 2: 5, 3: 5}
    out = bdg.compute_workers_per_gpu(9, gpus, scenes_per_gpu)
    assert out == {0: 3, 1: 2, 2: 2, 3: 2}


def test_compute_workers_per_gpu_clamp_to_scene_count():
    gpus = [0, 1, 2, 3]
    scenes_per_gpu = {0: 1, 1: 5, 2: 5, 3: 5}
    out = bdg.compute_workers_per_gpu(8, gpus, scenes_per_gpu)
    assert out[0] == 1  # clamped: only 1 scene on GPU 0
    assert out[1] == 2


def test_compute_workers_per_gpu_skips_empty_gpu():
    gpus = [0, 1]
    scenes_per_gpu = {0: 4, 1: 0}
    out = bdg.compute_workers_per_gpu(4, gpus, scenes_per_gpu)
    assert out == {0: 4}  # GPU 1 has no scenes -> skipped


def test_aggregate_summaries():
    summaries = [
        {"scene_id": "A", "processed": 3, "skipped": 1, "failed": [{"e": "x"}]},
        {"scene_id": "B", "processed": 2, "skipped": 0, "failed": [{"e": "y"}, {"e": "z"}]},
        {"scene_id": "C", "processed": 0, "skipped": 0, "failed": [{"e": "w"}]},
    ]
    agg = bdg.aggregate_summaries(summaries)
    assert agg["scenes_processed"] == 2  # A and B processed >=1
    assert agg["scenes_failed"] == 1  # C processed 0 and had failures
    assert agg["processed"] == 5
    assert agg["skipped"] == 1
    assert len(agg["failed"]) == 4


def test_clear_stale_failure_log_only_when_overwriting(tmp_path):
    failures_path = tmp_path / "failures.jsonl"
    failures_path.write_text('{"error":"old"}\n', encoding="utf-8")

    bdg.clear_stale_failure_log(tmp_path, overwrite=False)

    assert failures_path.exists()

    bdg.clear_stale_failure_log(tmp_path, overwrite=True)

    assert not failures_path.exists()


def test_multiprocess_arg_defaults():
    parser = bdg.build_arg_parser()
    args = parser.parse_args(["--subgoal_distance", "3.0"])
    assert args.num_processes == 1
    assert args.gpus == "0"


def test_multiprocess_args_parsed():
    parser = bdg.build_arg_parser()
    args = parser.parse_args(
        ["--subgoal_distance", "3.0", "--num_processes", "8", "--gpus", "0,1,2,3"]
    )
    assert args.num_processes == 8
    assert args.gpus == "0,1,2,3"
