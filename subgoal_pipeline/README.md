# R2R Landmark Sub-goal Pipeline

This package builds a compact R2R train split whose `goals` field is replaced
by ordered landmark sub-goals:

```json
"goals": [
  {"position": [x, y, z], "radius": 3.0},
  {"position": [x, y, z], "radius": 3.0}
]
```

The final sub-goal is always forced to the original `train.json.gz` goal position.

## Pipeline

1. Filter episodes whose GT action count is `<= --max_gt_actions`.
2. Group episodes by `trajectory_id`.
3. Pick the longest-instruction episode in each trajectory as the representative.
4. Replay the representative episode with GT actions in Habitat.
5. Ask GPT to select landmark-based keyframes from the replayed frames.
6. Save only compact artifacts:
   - `source_episode.json`
   - `subgoals_openai.json`
   - `subgoals_openai.mp4`
7. Reuse the representative trajectory sub-goals for sibling episodes.
8. Write the new dataset JSON.
9. Shift non-final sub-goals forward along the GT replay by a small step offset.
10. Regularize spacing and analyze distance distributions.

## Commands

Dry-run trajectory selection:

```bash
python3 -m subgoal_pipeline.build_dataset \
  --train_json VLN-CE/datasets/r2r/train/train.json.gz \
  --gt_json VLN-CE/datasets/r2r/train/train_gt.json.gz \
  --target_episodes 2000 \
  --max_gt_actions 80 \
  --dry_run_selection
```

Build the dataset:

```bash
export OPENAI_API_KEY=...
python3 -m subgoal_pipeline.build_dataset \
  --config rlinf/envs/habitat/extensions/config/vlnce_r2r_uninavid.yaml \
  --split train \
  --train_json VLN-CE/datasets/r2r/train/train.json.gz \
  --gt_json VLN-CE/datasets/r2r/train/train_gt.json.gz \
  --scenes_dir VLN-CE/scene_dataset \
  --target_episodes 2000 \
  --max_gt_actions 80 \
  --out_dir results/r2r_subgoal_online_train2000 \
  --output_json VLN-CE/datasets/r2r/train/r2r_train_with_subgoals.json.gz \
  --model gpt-5.4 \
  --overwrite
```

## Geodesic variant (no GPT)

`build_dataset_geodesic.py` selects sub-goals deterministically every `--subgoal_distance`
meters of `sim.geodesic_distance` along the replayed GT trajectory, drops a trailing
sub-goal that is closer than `D` to the final goal, and keeps the original episode final
goal. No OpenAI call, no frame sampling, no token usage. Spacing is regular by
construction, so `shift_subgoals.py` / `regularize_spacing.py` are not needed.

```bash
python3 -m subgoal_pipeline.build_dataset_geodesic \
  --config rlinf/envs/habitat/extensions/config/vlnce_rxr_uninavid.yaml \
  --split train \
  --train_json VLN-CE/datasets/rxr/train/train_guide_reachable.json.gz \
  --gt_json VLN-CE/datasets/rxr/train/train_guide_gt_reachable.json.gz \
  --scenes_dir VLN-CE/scene_dataset \
  --target_episodes 2000 --max_gt_actions 200 --min_gt_actions 150 \
  --out_dir results/rxr_subgoal_geodesic_train2000 \
  --output_json VLN-CE/datasets/rxr/train/train_guide_subgoals_geodesic.json.gz \
  --subgoal_distance 3.0 --overwrite
```

Shift non-final sub-goals by two replay steps:

```bash
python3 -m subgoal_pipeline.shift_subgoals \
  --out_dir results/r2r_subgoal_online_train2000 \
  --dataset_json VLN-CE/datasets/r2r/train/r2r_train_with_subgoals.json.gz \
  --original_train_json VLN-CE/datasets/r2r/train/train.json.gz \
  --step_offset 2
```

Regularize spacing:

```bash
python3 -m subgoal_pipeline.regularize_spacing \
  --out_dir results/r2r_subgoal_online_train2000 \
  --dataset_json VLN-CE/datasets/r2r/train/r2r_train_with_subgoals.json.gz \
  --original_train_json VLN-CE/datasets/r2r/train/train.json.gz \
  --close_threshold 1.0 \
  --far_threshold 6.0
```

Analyze:

```bash
python3 -m subgoal_pipeline.analyze_dataset \
  --dataset_json VLN-CE/datasets/r2r/train/r2r_train_with_subgoals.json.gz \
  --out_dir results/r2r_subgoal_analysis
```

## Files

- `build_dataset.py`: main online GPT pipeline.
- `build_dataset_geodesic.py`: GPT-free geodesic-distance sub-goal pipeline.
- `shift_subgoals.py`: move non-final sub-goals forward on the GT replay trajectory.
- `regularize_spacing.py`: remove close sub-goals and insert GT-trajectory midpoints.
- `analyze_dataset.py`: distance CSVs, summaries, and histograms.
- `gt.py`: `train_gt.json.gz` parsing and trajectory-group selection.
- `replay.py`: GT-action replay in Habitat memory.
- `openai_client.py`: GPT prompt, schema, API call, token usage parsing.
- `artifacts.py`: `source_episode.json`, `subgoals_openai.json`, and dataset goal conversion helpers.
- `video.py`: visualization video with red landmark text at sub-goal frames.
- `common.py`: JSON, gzip, and small shared helpers.
