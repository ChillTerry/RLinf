import math

from .replay import MemoryStep


def _geodesic(sim, a, b, *, episode_label: str = "", step_idx=None) -> float:
    raw = sim.geodesic_distance(a, b)
    try:
        d = float(raw)
    except (TypeError, ValueError):
        d = math.inf
    if math.isfinite(d) and d >= 0:
        return d
    fallback = math.dist(list(a), list(b))
    print(
        f"  !! geodesic_distance not finite (raw={raw!r}); "
        f"falling back to euclidean={fallback:.3f} "
        f"ep={episode_label} step={step_idx} a={list(a)} b={list(b)}"
    )
    return fallback


def select_geodesic_subgoal_steps(
    steps: list[MemoryStep],
    sim,
    subgoal_distance: float,
    final_goal_position: list[float],
    *,
    episode_label: str = "",
) -> list[int]:
    selected: list[int] = []
    if len(steps) < 2:
        return selected
    anchor = steps[0].agent_position
    for i in range(1, len(steps)):
        d = _geodesic(
            sim, anchor, steps[i].agent_position,
            episode_label=episode_label, step_idx=i,
        )
        if d >= float(subgoal_distance):
            selected.append(i)
            anchor = steps[i].agent_position
    while selected:
        last_step = steps[selected[-1]]
        d = _geodesic(
            sim, last_step.agent_position, final_goal_position,
            episode_label=episode_label, step_idx=last_step.step,
        )
        if d < float(subgoal_distance):
            selected.pop()
        else:
            break
    return selected
