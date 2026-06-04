# Subgoal Reward TensorBoard Metrics

本文档定义 `subgoal_progress` reward 需要加入 TensorBoard 的监控变量。所有变量均用于诊断 reward 设计，不参与 reward 计算。

## 聚合语义

当前 env metrics 通过 `infos["episode"]` 进入 TensorBoard。每个 key 在一个 rollout 内按 finished episodes 求均值。因此，以下变量应优先设计成 episode-level scalar。

对于条件比例，例如“发生过 stall penalty 的 episode 中最终成功到达 goal 的比例”，不能直接把 no-stall episode 填 0 或 1，否则 TensorBoard 均值会改变数学含义。此类指标采用 numerator / denominator 两个变量记录：

```text
conditional_ratio = mean(numerator) / mean(denominator)
```

## 指标定义

### env/subgoal_completion_ratio

```text
completed_subgoal_count / num_subgoals
```

表示一个 episode 最终完成了多少比例的 subgoals。这里的完成标准使用 `subgoal_switch_distance`，即当前设计中的 1.0m 切换阈值。

该指标衡量粗粒度 waypoint 推进能力。如果该值上升，说明 agent 更频繁地沿 subgoal 序列向前推进。

### env/precision_subgoal_success_ratio

```text
precision_subgoal_success_count / num_intermediate_subgoals
```

表示 intermediate subgoals 中有多少比例真正达到 `subgoal_success_distance`，即当前设计中的 0.5m 精确成功阈值。

该指标比 `env/subgoal_completion_ratio` 更严格。它用于区分 agent 是仅仅进入 1.0m 切换范围，还是确实接近 subgoal 到 0.5m 以内。

### env/stall_penalty_rate

```text
stall_penalty_step_count / valid_reward_step_count
```

表示一个 episode 的有效 reward steps 中，有多少比例触发了 stall penalty。

该指标衡量 step-level 卡住强度。如果该值长期偏高，说明 agent 经常没有朝 active subgoal 取得正向 progress，或者 `stall_patience` 设置过小。

### env/any_stall_penalty_subgoal

```text
1.0 if any subgoal triggered stall penalty at least once else 0.0
```

表示一个 episode 内是否存在至少一个 subgoal 触发过 stall penalty。

这是计算 stall 后恢复成功率的 denominator。

### env/stall_then_goal_success

```text
1.0 if any_stall_penalty_subgoal and final_goal_success else 0.0
```

表示一个 episode 发生过 stall penalty 后，最终是否仍然成功到达 final goal。

这是计算 stall 后恢复成功率的 numerator。

### derived/stall_then_goal_success_ratio

```text
mean(env/stall_then_goal_success) / mean(env/any_stall_penalty_subgoal)
```

表示在发生过 stall penalty 的 episodes 中，最终成功到达 final goal 的比例。

该指标不应直接作为普通 per-episode scalar 写入 TensorBoard，因为 no-stall episodes 的分母为 0。应使用 `env/stall_then_goal_success` 和 `env/any_stall_penalty_subgoal` 两条曲线计算，或后续在聚合阶段专门生成 derived metric。

### env/first_stall_penalty_rate

```text
first_stall_penalty_subgoal_count / num_subgoals
```

表示一个 episode 中有多少比例的 subgoals 至少触发过一次 stall penalty。每个 subgoal 最多计数一次。

该指标衡量 subgoal-level 卡住覆盖率。它与 `env/stall_penalty_rate` 的区别是：`env/stall_penalty_rate` 会被同一个 subgoal 的连续卡住步数放大，而 `env/first_stall_penalty_rate` 只关心有多少 subgoals 曾经卡住过。

### env/premature_stop_ratio

```text
1.0 if stop_action and not all_subgoals_finished_before_stop else 0.0
```

表示一个 episode 是否发生 premature stop。

TensorBoard 中该值的均值就是 premature stop episode ratio。它用于判断 policy 是否过早输出 stop action。

### env/final_goal_success_ratio

```text
1.0 if stop_action and distance_to_final_goal <= final_success_distance else 0.0
```

表示一个 episode 是否在 stop 时成功到达 final goal。当前设计中的 `final_success_distance` 为 3.0m。

该指标是最终任务成功率的 reward-side 诊断变量，用于和 `env/stall_then_goal_success`、`env/premature_stop_ratio` 配合分析 stop 行为。

### env/distance_to_final_goal

```text
geodesic distance from current agent position to final goal
```

表示 episode 结束时 agent 到 final goal 的 geodesic distance。

该指标是连续值，比 success ratio 更早反映训练趋势。即使 success 尚未明显提升，只要该距离下降，也说明导航行为可能在改善。

### env/stop_action_ratio

```text
1.0 if terminal action is stop else 0.0
```

表示一个 episode 是否由 stop action 结束。

该指标用于解释 `env/premature_stop_ratio` 和 `env/final_goal_success_ratio`。例如 premature stop 下降可能来自 policy 更少 stop，而不一定来自导航更好，因此需要同时观察 stop action ratio。

### env/mean_normalized_progress

```text
cumulative_normalized_progress / valid_reward_step_count
```

表示一个 episode 内每个有效 reward step 的平均 normalized progress。

该指标衡量 dense progress signal 是否持续为正。如果该值长期接近 0 或为负，说明 active subgoal progress 对 policy 的训练信号较弱或方向不稳定。

### env/num_subgoals

```text
number of subgoals in the episode
```

表示一个 episode 的 subgoal 数量。

该指标不参与 subgoal ratio 的归一化，因为相关 subgoal 指标本身已经是 ratio。它主要用于解释 batch composition 和 cumulative reward scale：如果某段训练中平均 `num_subgoals` 变多，episode 可能更难，`cumulative_progress_reward`、`cumulative_penalty_reward` 等非 ratio 指标也可能随之变化。

### env/cumulative_progress_reward

```text
sum(r_progress over valid reward steps)
```

表示一个 episode 中累计获得的 progress reward。

该指标用于判断 dense progress reward 对总 reward 的贡献是否稳定，以及是否随 `num_subgoals` 或 episode difficulty 出现系统性偏移。

### env/cumulative_subgoal_success_reward

```text
sum(r_subgoal_success over valid reward steps)
```

表示一个 episode 中累计获得的 intermediate subgoal success bonus。

由于 final subgoal 不再给 `r_subgoal_success`，该指标只反映 intermediate subgoal 的精确到达奖励。

### env/cumulative_penalty_reward

```text
sum(r_penalty over valid reward steps)
```

表示一个 episode 中累计获得的 stall penalty reward。

该值通常为非正数。它用于判断 penalty 是否过强，以及是否在训练中长期主导总 reward。

### env/cumulative_stop_reward

```text
sum(r_stop over valid reward steps)
```

表示一个 episode 中累计获得的 stop reward。

该指标包含 premature stop penalty 和 final stop success reward。它用于分析 stop action 的 reward 贡献是否与 `env/final_goal_success_ratio`、`env/premature_stop_ratio` 一致。
