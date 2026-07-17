import pytest
from omegaconf import OmegaConf

from rlinf.config import (
    SupportedModel,
    validate_habitat_uninavid_curriculum_cfg,
)
from rlinf.envs.habitat.extensions.bucket_scheduler import (
    normalize_bucket_curriculum_plan,
)


def _config(train_overrides=None):
    train = {
        "env_type": "habitat",
        "action_length_bucketing": True,
        "min_gt_action_length": 20,
        "max_gt_action_length": 80,
        "gt_path": "/tmp/gt.json.gz",
        "bucket_step_range_map": {
            "short": [20, 40],
            "medium": [40, 60],
            "long": [60, 80],
        },
        "bucket_max_steps": [40, 80, 120],
        "bucket_schedule_seed": 42,
        "curriculum_interval": 50,
        "curriculum_stages_map": {
            "stage_a": [2, 1, 1],
            "stage_b": [1, 1, 2],
        },
    }
    train.update(train_overrides or {})
    return OmegaConf.create(
        {
            "env": {"train": train},
            "algorithm": {"rollout_epoch": 4, "adv_type": "gae"},
            "actor": {
                "seed": 42,
                "training_backend": "fsdp",
                "model": {"model_type": "uninavid", "num_action_chunks": 4},
            },
            "runner": {
                "weight_sync_interval": 1,
                "overlap_env_bootstrap": False,
            },
        }
    )


def _normalize(**overrides):
    values = {
        "bucket_step_range_map": {"a": [20, 40], "b": [40, 60]},
        "bucket_max_steps": [40, 80],
        "curriculum_stages_map": {"stage": [2, 2]},
        "rollout_epoch": 4,
        "curriculum_interval": 50,
        "effective_num_action_chunks": 4,
        "bucket_schedule_seed": 42,
    }
    values.update(overrides)
    return normalize_bucket_curriculum_plan(**values)


def test_feature_off_does_not_require_curriculum_fields():
    cfg = OmegaConf.create({"env": {"train": {"env_type": "libero"}}, "algorithm": {}})

    validate_habitat_uninavid_curriculum_cfg(cfg, SupportedModel.OPENPI)

    assert cfg.env.train.action_length_bucketing is False


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"bucket_step_range_map": {}}, "at least one"),
        ({"bucket_step_range_map": {"a": [20]}}, "two integers"),
        ({"bucket_step_range_map": {"a": [20, 20]}}, "lower < upper"),
        (
            {"bucket_step_range_map": {"a": [20, 50], "b": [40, 60]}},
            "overlapping",
        ),
        ({"bucket_max_steps": [40]}, "length"),
        ({"bucket_max_steps": [40, 0]}, "positive"),
        ({"bucket_max_steps": [40, 82]}, "not divisible"),
        ({"curriculum_stages_map": {}}, "at least one"),
        ({"curriculum_stages_map": {"stage": [4]}}, "quota length"),
        ({"curriculum_stages_map": {"stage": [5, -1]}}, "non-negative"),
        ({"curriculum_stages_map": {"stage": [1, 1]}}, "sum to"),
        ({"curriculum_interval": 0}, "positive"),
    ],
)
def test_invalid_explicit_plan_fails(overrides, message):
    with pytest.raises(ValueError, match=message):
        _normalize(**overrides)


def test_non_adjacent_declared_overlap_is_rejected_but_gap_is_valid():
    with pytest.raises(ValueError, match="overlapping"):
        _normalize(
            bucket_step_range_map={
                "first": [20, 30],
                "second": [50, 60],
                "third": [25, 40],
            },
            bucket_max_steps=[40, 80, 120],
            curriculum_stages_map={"stage": [2, 1, 1]},
        )

    plan = _normalize(bucket_step_range_map={"a": [20, 30], "b": [40, 60]})
    assert plan["bucket_ids"] == ["a", "b"]


def test_enabled_feature_injects_actual_train_chunk_count_and_rollout_epoch():
    cfg = _config()

    validate_habitat_uninavid_curriculum_cfg(cfg, SupportedModel.UNINAVID)

    assert cfg.env.train.effective_num_action_chunks == 4
    assert cfg.env.train.rollout_epoch == 4
    assert cfg.env.train.max_episode_steps == 120
    assert cfg.env.train.max_steps_per_rollout_epoch == 120


def test_feature_rejects_non_habitat_or_non_uninavid():
    with pytest.raises(ValueError, match=r"Habitat \+ UniNaVid"):
        validate_habitat_uninavid_curriculum_cfg(
            _config({"env_type": "libero"}), SupportedModel.UNINAVID
        )
    with pytest.raises(ValueError, match=r"Habitat \+ UniNaVid"):
        validate_habitat_uninavid_curriculum_cfg(_config(), SupportedModel.OPENPI)


def test_curriculum_rejects_async_weight_versions_and_fixed_horizon_prefetch():
    cfg = _config()
    cfg.runner.weight_sync_interval = 2
    with pytest.raises(ValueError, match="weight_sync_interval=1"):
        validate_habitat_uninavid_curriculum_cfg(cfg, SupportedModel.UNINAVID)

    cfg = _config()
    cfg.runner.overlap_env_bootstrap = True
    with pytest.raises(ValueError, match="bootstrap prefetch"):
        validate_habitat_uninavid_curriculum_cfg(cfg, SupportedModel.UNINAVID)
