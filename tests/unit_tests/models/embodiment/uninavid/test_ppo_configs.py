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


def test_habitat_rxr_ppo_uninavid_config_selects_gae_actor_critic():
    cfg = _load_config("habitat_rxr_ppo_uninavid.yaml")

    assert cfg.algorithm.adv_type == "gae"
    assert cfg.algorithm.loss_type == "actor_critic"
    assert cfg.algorithm.group_size == 1
    assert cfg.actor.model.add_value_head is True
    assert cfg.critic.use_critic_model is False
