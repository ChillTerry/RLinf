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
python3 -m streamvln.subgoal_pipeline.build_dataset \
  --train_json R2R_VLNCE_v1-3_preprocessed/train/train.json.gz \
  --gt_json R2R_VLNCE_v1-3_preprocessed/train/train_gt.json.gz \
  --target_episodes 2000 \
  --max_gt_actions 80 \
  --dry_run_selection
```

Build the dataset:

```bash
export OPENAI_API_KEY=...
python3 -m streamvln.subgoal_pipeline.build_dataset \
  --config config/vln_r2r.yaml \
  --split train \
  --train_json R2R_VLNCE_v1-3_preprocessed/train/train.json.gz \
  --gt_json R2R_VLNCE_v1-3_preprocessed/train/train_gt.json.gz \
  --target_episodes 2000 \
  --max_gt_actions 80 \
  --out_dir results/r2r_subgoal_online_train2000 \
  --output_json R2R_VLNCE_v1-3_preprocessed/train/r2r_train_with_subgoals.json \
  --model gpt-5.4 \
  --overwrite
```

Shift non-final sub-goals by two replay steps:

```bash
python3 -m streamvln.subgoal_pipeline.shift_subgoals \
  --out_dir results/r2r_subgoal_online_train2000 \
  --dataset_json R2R_VLNCE_v1-3_preprocessed/train/r2r_train_with_subgoals.json \
  --original_train_json R2R_VLNCE_v1-3_preprocessed/train/train.json.gz \
  --step_offset 2
```

Regularize spacing:

```bash
python3 -m streamvln.subgoal_pipeline.regularize_spacing \
  --out_dir results/r2r_subgoal_online_train2000 \
  --dataset_json R2R_VLNCE_v1-3_preprocessed/train/r2r_train_with_subgoals.json \
  --original_train_json R2R_VLNCE_v1-3_preprocessed/train/train.json.gz \
  --close_threshold 1.0 \
  --far_threshold 6.0
```

Analyze:

```bash
python3 -m streamvln.subgoal_pipeline.analyze_dataset \
  --dataset_json R2R_VLNCE_v1-3_preprocessed/train/r2r_train_with_subgoals.json \
  --out_dir results/r2r_subgoal_analysis
```

## Files

- `build_dataset.py`: main online GPT pipeline.
- `shift_subgoals.py`: move non-final sub-goals forward on the GT replay trajectory.
- `regularize_spacing.py`: remove close sub-goals and insert GT-trajectory midpoints.
- `analyze_dataset.py`: distance CSVs, summaries, and histograms.
- `gt.py`: `train_gt.json.gz` parsing and trajectory-group selection.
- `replay.py`: GT-action replay in Habitat memory.
- `openai_client.py`: GPT prompt, schema, API call, token usage parsing.
- `artifacts.py`: `source_episode.json`, `subgoals_openai.json`, and dataset goal conversion helpers.
- `video.py`: visualization video with red landmark text at sub-goal frames.
- `common.py`: JSON, gzip, and small shared helpers.
