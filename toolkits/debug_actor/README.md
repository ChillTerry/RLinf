# Debug actor training with cached rollout batches

This toolkit lets you debug the actor-training path without paying the full
environment + rollout cost on every run.

It adds two standalone scripts:

- `save_rollout_batch.py`: run one normal env→rollout step, capture each
  actor worker's `rollout_batch`, save it to disk, then exit before actor
  training starts.
- `replay_actor_training.py`: load the saved `rollout_batch`, launch only the
  actor workers, and run `compute_advantages_and_returns()` plus
  `run_training()` directly.

## Scope

- Supported: synchronous embodied PPO / GRPO paths using
  `EmbodiedRunner` + `EmbodiedFSDPActor`
- Not supported: SAC configs (`algorithm.loss_type=embodied_sac`)

## Prerequisites

1. Use the same RLinf environment you use for normal embodied training.
2. Make sure Ray / CUDA / model checkpoints / simulator dependencies are set up
   exactly as they would be for your normal config.
3. Use a config that already works with `examples/embodiment/run_embodiment.sh`
   or `examples/embodiment/train_embodied_agent.py`.

## 1) Capture a rollout batch once

Run the capture script with the same config you normally train with:

```bash
python toolkits/debug_actor/save_rollout_batch.py \
  --config-path ../../examples/embodiment/config \
  --config-name <config_name> \
  +save_dir=/tmp/debug_actor_cache
```

What it does:

1. launches env, rollout, and actor workers
2. performs one normal env interaction + rollout generation step
3. calls `recv_rollout_trajectories()` on the actor
4. saves each actor rank's processed `rollout_batch` to disk
5. writes the resolved Hydra config to `config.yaml`
6. exits before actor training starts

Expected outputs in `save_dir`:

- `rollout_batch_rank0.pt`
- `rollout_batch_rank1.pt`
- ...
- `config.yaml`

The exact number of `rollout_batch_rank*.pt` files must match the actor worker
count for your config.

## 2) Replay actor training quickly

After capture succeeds, replay actor training from the cached batch:

```bash
python toolkits/debug_actor/replay_actor_training.py \
  --config-path ../../examples/embodiment/config \
  --config-name <config_name> \
  +load_dir=/tmp/debug_actor_cache \
  +num_train_steps=1
```

What it does:

1. launches only the actor workers
2. initializes model / optimizer state with `init_worker()`
3. loads `rollout_batch_rank<R>.pt` into each actor rank
4. runs `compute_advantages_and_returns()`
5. runs `run_training()`

If `num_train_steps > 1`, the script reloads the saved batch between steps so
each replay iteration starts from the same cached input.

## Quick-start flow

```bash
# Step 1: cache one rollout batch
python toolkits/debug_actor/save_rollout_batch.py \
  --config-path ../../examples/embodiment/config \
  --config-name habitat_r2r_ppo_navid \
  +save_dir=/tmp/debug_actor_cache

# Step 2: debug actor-side code only
python toolkits/debug_actor/replay_actor_training.py \
  --config-path ../../examples/embodiment/config \
  --config-name habitat_r2r_ppo_navid \
  +load_dir=/tmp/debug_actor_cache \
  +num_train_steps=1
```

## Debugging tips

- Put `breakpoint()` in `rlinf/workers/actor/fsdp_actor_worker.py` inside
  `compute_advantages_and_returns()` or `run_training()`.
- For interactive stepping, prefer a single-node / single-GPU config.
- If you change rollout-side logic, environment outputs, or any preprocessing
  that affects `rollout_batch`, re-run the capture step.
- If you only change actor-side logic, you can usually skip re-capture and keep
  replaying the same cached batch.

## Common failure modes

- `rollout_batch_rank<R>.pt` not found:
  - the capture step did not run successfully, or
  - `load_dir` does not match the directory used during capture, or
  - actor world size changed between capture and replay.
- Replay starts but shape / key mismatches appear:
  - your code changed the expected `rollout_batch` structure; capture a fresh
    batch with the new code.
- SAC config error:
  - these scripts intentionally reject SAC because the workflow here is scoped
    to sync PPO / GRPO actor training.
