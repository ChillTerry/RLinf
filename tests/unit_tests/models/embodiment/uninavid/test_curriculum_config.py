import pytest
from omegaconf import OmegaConf

from rlinf.config import (
    SupportedModel,
    validate_habitat_uninavid_curriculum_cfg,
)


def _config(train_overrides=None):
    train = {
        "env_type": "habitat",
        "action_length_bucketing": True,
        "action_length_bin_size": 32,
        "max_gt_action_length": 96,
        "max_steps_ratio": 1.3,
        "gt_path": "/tmp/gt.json.gz",
        "bucket_curriculum_enabled": True,
        "curriculum_stages": [
            {"active_bucket_count": 1, "weights": [1.0]},
            {"active_bucket_count": 2, "weights": [0.75, 0.25]},
        ],
    }
    train.update(train_overrides or {})
    return OmegaConf.create(
        {
            "env": {"train": train},
            "algorithm": {"rollout_epoch": 2, "adv_type": "gae"},
            "actor": {
                "seed": 42,
                "training_backend": "fsdp",
                "model": {"num_action_chunks": 4},
            },
            "runner": {
                "weight_sync_interval": 1,
                "overlap_env_bootstrap": False,
            },
        }
    )


def test_feature_off_does_not_require_curriculum_fields():
    cfg = OmegaConf.create({"env": {"train": {"env_type": "libero"}}, "algorithm": {}})

    validate_habitat_uninavid_curriculum_cfg(cfg, SupportedModel.OPENPI)

    assert cfg.env.train.action_length_bucketing is False


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"action_length_bin_size": 0}, "action_length_bin_size"),
        ({"max_steps_ratio": 0}, "max_steps_ratio"),
        ({"gt_path": None}, "requires gt_path"),
        (
            {"curriculum_stages": [{"active_bucket_count": 2, "weights": [0.5, 0.4]}]},
            "sum to 1",
        ),
        (
            {
                "curriculum_stages": [
                    {"active_bucket_count": 3, "weights": [0.5, 0.3, 0.2]}
                ]
            },
            "exceeding rollout_epoch",
        ),
    ],
)
def test_invalid_curriculum_config_fails(overrides, message):
    with pytest.raises(ValueError, match=message):
        validate_habitat_uninavid_curriculum_cfg(
            _config(overrides), SupportedModel.UNINAVID
        )


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
