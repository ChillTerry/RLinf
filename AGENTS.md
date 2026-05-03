# AGENTS.md

## Change scope preference

- Prefer not to modify RLinf core framework code when working on this repo.
- Keep changes scoped to `rlinf/models/embodiment/` and `rlinf/envs/habitat/` whenever feasible.
- If a shared-path change is unavoidable, add it behind an explicit model-specific branch (prefer checking the model name / `model_type`) so the default execution path for other VLA models stays unchanged.
- Do not push model- or Habitat-specific behavior into the generic system flow unless that logic is isolated behind an opt-in condition for the relevant model or env.
- When touching shared code, document why a local change under `rlinf/models/embodiment/` or `rlinf/envs/habitat/` was insufficient and note the isolation mechanism used to avoid regressions in other models.
- Avoid degradation handling, fallback, hacks, heuristics, local stabilizations, or post-processing bandages that are not faithful general algorithms.
- Do not install any libraries without permission or make any changes that could damage the environment. If you encounter any environment-related errors, stop immediately and report them.
- When referring to code, do not use line-number references such as xxx.py:156. Just provide me with the source code directly.

---

## Code structure

- `**.cursor/**` – Rules and skills: `rules/agents-md.mdc`, `skills/add-install-docker-ci-e2e`, `skills/add-example-doc-model-env`, `skills/review-pr`.
- `**rlinf/**` – Main package:
  - `agents/` – Agent logic (reasoning, tools).
  - `algorithms/` – Advantages, losses, registry, rewards (math, code, searchr1, vqa).
  - `config.py` – Hydra config, `SupportedModel`, `SupportedEnvType`, validation.
  - `data/` – Datasets for embodied, reasoning, agent.
  - `envs/` – ManiSkill, LIBERO, IsaacLab, CALVIN, MetaWorld, Behavior, RoboCasa, FrankaSim, RealWorld, RoboTwin, Habitat, OpenSora world model; `get_env_cls()` in `envs/__init__.py`.
  - `hybrid_engines/` – SGLang/vLLM rollout integration.
  - `models/` – Embodiment (OpenVLA, OpenVLA-OFT, OpenPI, GR00T, MLP/CNN/Flow/CMA) and reasoning wiring.
  - `runners/` – Embodied (sync/async), reasoning, coding_online_rl, agent, SFT, eval.
  - `scheduler/` – Cluster, Worker, WorkerGroup, channel, manager, placement, dynamic_scheduler.
  - `utils/` – Logging, placement, data iter, distributed, checkpoint, resharding.
  - `workers/` – Actor (FSDP/Megatron), rollout (HF/server), env (sync/async), reward, replay buffer.
- `**examples/**` – Entrypoints and YAML: embodiment, reasoning, coding_online_rl, searchr1, sft, wideseek_r1.
- `**tests/**` – `unit_tests/`, `e2e_tests/` (embodied, agent, reasoning), scheduler tests; e2e configs under `e2e_tests/embodied/*.yaml`.
- `**requirements/**` – `install.sh` (targets: embodied, reason, docs; `--model`, `--env`), optional deps in subdirs.
- `**docker/**` – Dockerfile and build targets per model/env.
- `**ray_utils/**` – `start_ray.sh` (multi-node head/worker), `check_ray.sh`, `realworld/setup_before_ray.sh`.
- `**toolkits/**` – Checkpoint convertors, verifiers, eval scripts, replay buffer, auto-placement.
- `**docs/**` – Sphinx RST (EN/ZH): start, tutorials, examples, APIs, FAQ.

---

## How RLinf runs

You launch one entry script (e.g. `train_embodied_agent.py`, `train_async.py`). It builds a **Cluster** (Ray must already be up), figures **component placement** (actor, rollout, env, reward, agent), and starts **Worker** groups. A **Runner** drives the loop: rollout → reward → advantage → actor update (and any inference/engine lifecycle). Cluster config lives in YAML under `cluster:`: `num_nodes`, `component_placement`, `node_groups` (labels, node_ranks, env_configs, optional hardware e.g. Franka). Placement (e.g. `HybridComponentPlacement`, `ModelParallelComponentPlacement`) maps components to node groups and hardware ranks. Workers are Ray remote actors with `MASTER_`*, `RANK`, etc.; they can `send`/`recv` across groups. Training backends: FSDP or Megatron. Rollout: SGLang or vLLM. Runners pick loss/advantage from config (PPO, GRPO, SAC, etc.).

---

## Key ideas and plugging in

**Config** (`rlinf/config.py`): `build_config` / `validate_cfg` produce the full DictConfig. New model or env types go into `SupportedModel` / `SupportedEnvType` and validation.

**Cluster and placement:** `ClusterConfig` and strategies in `rlinf/scheduler/placement/`, `rlinf/utils/placement.py`. Placement controls where actor/rollout/env run (one node vs many, GPU vs CPU, heterogeneous).

**Algorithms:** Advantage and loss functions are registered in `rlinf/algorithms/` (registry + decorators); rewards are registered in `rlinf/algorithms/rewards/`. Config keys `algorithm.adv_type` and `algorithm.loss_type` select them. See [Extending RLinf: algorithms, models, envs](#extending-rlinf-algorithms-models-envs) for step-by-step instructions.

**Models (embodied):** Register in `SupportedModel` in `config.py`, implement under `rlinf/models/embodiment/<name>/` (e.g. `BasePolicy`), wire in config and workers. Use add-install-docker-ci-e2e for install/Docker/CI. Details in the extension section below.

**Environments:** Register in `SupportedEnvType` and `get_env_cls()` in `rlinf/envs/__init__.py`, implement under `rlinf/envs/<name>/`. Use add-install-docker-ci-e2e and add-example-doc-model-env for install and docs. Details below.

**Workers:** Subclass `Worker`, implement `initialize` and your API, launch with `create_group(...).launch(...)`. Use `self.log_info` / `log_warning` / `log_error`; no print.

**Runners:** They own the training loop. New task type = new runner + entry script that builds Cluster, placement, worker groups, and calls the runner.

---

## Extending RLinf: algorithms, models, envs

**Advantage function**

- Implement a function that takes the same keyword args as existing ones (e.g. `rewards`, `values`, `dones`, `gamma`, `loss_mask`, …) and returns `(advantages, returns)`. See `rlinf/algorithms/advantages.py` (e.g. `compute_gae_advantages_and_returns`) for signatures.
- Register it: `from rlinf.algorithms.registry import register_advantage` then `@register_advantage("my_adv")` on your function. The name is case-normalized to lowercase.
- In config YAML set `algorithm.adv_type: my_adv`. Actor workers call `calculate_adv_and_returns(adv_type=...)` which dispatches via `get_adv_and_returns(name)`.
- For non-GAE styles (e.g. GRPO, Reinforce++), `rlinf/algorithms/utils.py` may need to compute scores first; check how `adv_type` is used in `calculate_adv_and_returns` and in the actor worker.

**Policy loss**

- Implement a function that accepts the kwargs passed by the actor (e.g. `logprobs`, `old_logprobs`, `advantages`, `clip_ratio_low`, `clip_ratio_high`, `loss_mask`, …) and returns `(loss_tensor, metrics_dict)`. See `rlinf/algorithms/losses.py` (e.g. `compute_ppo_actor_loss`, `compute_ppo_actor_critic_loss`, `compute_grpo_actor_loss_fn`).
- Register: `from rlinf.algorithms.registry import register_policy_loss` then `@register_policy_loss("my_loss")`.
- In config set `algorithm.loss_type: my_loss`. For PPO-style actor+critic you need a critic and value loss; the unified entry is `policy_loss(loss_type=..., **kwargs)` in `registry.py`. Add validation in `rlinf/config.py` if your loss has special requirements (e.g. `validate_cfg` already checks `loss_type == "actor_critic"` for value head).

**Reward**

- Add a reward class (e.g. under `rlinf/algorithms/rewards/<domain>/`) that matches the interface expected by the reward worker (e.g. callable or class with a clear contract for prompt/completions/ids).
- In `rlinf/algorithms/rewards/__init__.py`: import the class, then `register_reward("my_reward", MyRewardClass)`. The registry is `reward_registry`; lookup via `get_reward_class(name)`.
- Wire the reward name in config and in the runner/reward worker so the correct class is instantiated and used. For reasoning/agent tasks the config path may be under `reward.path` or similar.
