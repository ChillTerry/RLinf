from pathlib import Path

from omegaconf import OmegaConf


def _load_config(name: str):
    return OmegaConf.load(Path("examples/embodiment/config") / name)


def test_habitat_r2r_ppo_uninavid_config_selects_gae_actor_critic():
    cfg = _load_config("habitat_r2r_ppo_uninavid.yaml")
    model_cfg = _load_config("model/uninavid.yaml")

    assert cfg.algorithm.adv_type == "gae"
    assert cfg.algorithm.loss_type == "actor_critic"
    assert cfg.algorithm.logprob_type == "chunk_level"
    assert cfg.algorithm.group_size == 1
    assert cfg.actor.model.add_value_head is True
    assert model_cfg.detach_critic_input is True
    assert cfg.critic.use_critic_model is False
    assert cfg.env.train.action_length_bucketing is False


def test_habitat_rxr_ppo_uninavid_config_selects_gae_actor_critic():
    cfg = _load_config("habitat_rxr_ppo_uninavid.yaml")

    assert cfg.algorithm.adv_type == "gae"
    assert cfg.algorithm.loss_type == "actor_critic"
    assert cfg.algorithm.group_size == 1
    assert cfg.actor.model.add_value_head is True
    assert cfg.critic.use_critic_model is False


def test_habitat_rxr_grpo_uninavid_config_exposes_opt_in_curriculum():
    cfg = _load_config("habitat_rxr_grpo_uninavid.yaml")

    assert cfg.env.train.action_length_bucketing is True
    assert cfg.algorithm.rollout_epoch == 8
    assert list(cfg.env.train.bucket_step_range_map) == [
        "bucket_1",
        "bucket_2",
        "bucket_3",
        "bucket_4",
    ]
    assert list(cfg.env.train.curriculum_stages_map.stages_1) == [5, 3, 0, 0]
