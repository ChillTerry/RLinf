import subprocess
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

from .replay import MemoryStep


class VideoWriter:
    def __init__(self, output_path: Path, width: int, height: int, fps: int) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{int(width)}x{int(height)}",
            "-pix_fmt",
            "rgb24",
            "-r",
            str(int(fps)),
            "-i",
            "-",
            "-an",
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(self.output_path),
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write_rgb(self, frame: np.ndarray) -> None:
        if self.proc.stdin is None:
            raise RuntimeError("ffmpeg stdin is closed.")
        self.proc.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())

    def close(self) -> None:
        if self.proc.stdin is not None:
            self.proc.stdin.close()
        ret = self.proc.wait()
        if ret != 0:
            raise RuntimeError(f"ffmpeg exited with code {ret}: {self.output_path}")


def _wrap_lines(
    text: str,
    *,
    font: int,
    font_scale: float,
    thickness: int,
    max_width: int,
) -> List[str]:
    """Wrap text so each line fits within max_width measured by cv2."""
    words = str(text).split()
    if not words:
        return [str(text)]

    lines: List[str] = []
    cur = ""
    for word in words:
        trial = word if not cur else f"{cur} {word}"
        (width, _), _ = cv2.getTextSize(trial, font, font_scale, thickness)
        if width <= max_width or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def draw_center_highlight(
    frame: np.ndarray,
    text: str,
    *,
    box_alpha: float = 0.6,
    padding_x: int = 22,
    padding_y: int = 16,
    line_spacing: int = 8,
) -> np.ndarray:
    """Draw a centered, darkened highlight box with wrapped text using cv2 builtin font."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    font_scale = max(0.6, frame.shape[0] / 1600.0)
    max_width = int(frame.shape[1] * 0.82)

    lines = _wrap_lines(
        text,
        font=font,
        font_scale=font_scale,
        thickness=thickness,
        max_width=max_width,
    )

    line_sizes = [
        cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines
    ]
    text_width = max((w for w, _ in line_sizes), default=0)
    text_height = sum(h for _, h in line_sizes) + line_spacing * max(0, len(lines) - 1)
    box_w = text_width + padding_x * 2
    box_h = text_height + padding_y * 2
    x0 = (frame.shape[1] - box_w) // 2
    y0 = (frame.shape[0] - box_h) // 2

    frame = frame.copy()
    roi = frame[y0 : y0 + box_h, x0 : x0 + box_w]
    if roi.size:
        cv2.addWeighted(roi, 1.0 - box_alpha, roi, 0.0, 0.0, roi)

    y = y0 + padding_y
    for (line, (_w, h)) in zip(lines, line_sizes):
        text_y = y + h
        cv2.putText(
            frame,
            line,
            (x0 + padding_x, text_y),
            font,
            font_scale,
            (0, 0, 0),
            thickness + 2,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            line,
            (x0 + padding_x, text_y),
            font,
            font_scale,
            (255, 40, 40),
            thickness,
            lineType=cv2.LINE_AA,
        )
        y += h + line_spacing
    return frame


def write_subgoal_video_from_memory(
    output_path: Path,
    steps: List[MemoryStep],
    subgoals: List[dict],
    fps: int,
    highlight_frames: int,
    *,
    topdown_metric: Any = None,
    sim: Any = None,
    instruction: str = "",
    subgoal_radius: float = 0.0,
) -> None:
    if not steps:
        return

    base = None
    grid_resolution = None
    if topdown_metric is not None and sim is not None:
        base, grid_resolution, _meters_per_row, _meters_per_col = _build_topdown_base(
            topdown_metric, subgoals, sim, subgoal_radius
        )

    highlight_by_step: Dict[int, str] = {
        int(item["best_step"]): str(item.get("landmark") or "")
        for item in subgoals
    }
    highlight_frames = max(1, int(highlight_frames))
    highlight_remaining = 0
    active_landmark = ""

    writer: Any = None
    try:
        for step_info in steps:
            if int(step_info.step) in highlight_by_step:
                active_landmark = highlight_by_step[int(step_info.step)]
                highlight_remaining = highlight_frames

            frame = np.asarray(step_info.rgb, dtype=np.uint8)
            if highlight_remaining > 0 and active_landmark:
                frame = draw_center_highlight(frame, active_landmark)

            if base is not None:
                from habitat.utils.visualizations import maps as habitat_maps
                from rlinf.envs.habitat.extensions.video import append_instruction_panel

                td = base.copy()
                pos = step_info.agent_position
                row, col = habitat_maps.to_grid(
                    float(pos[2]), float(pos[0]), grid_resolution, sim=sim
                )
                cv2.circle(td, (int(col), int(row)), 6, (255, 0, 0), -1)
                cv2.circle(td, (int(col), int(row)), 6, (255, 255, 255), 1, lineType=cv2.LINE_AA)
                td = cv2.resize(
                    td,
                    (int(frame.shape[1]), int(frame.shape[0])),
                    interpolation=cv2.INTER_LINEAR,
                )
                frame = np.concatenate((frame, td.astype(np.uint8)), axis=1)
                frame = append_instruction_panel(frame, instruction)

            if writer is None:
                writer = VideoWriter(
                    output_path=output_path,
                    width=int(frame.shape[1]),
                    height=int(frame.shape[0]),
                    fps=int(fps),
                )
            writer.write_rgb(frame)

            if highlight_remaining > 0:
                highlight_remaining -= 1
                if highlight_remaining == 0:
                    active_landmark = ""
    finally:
        if writer is not None:
            writer.close()


def _build_topdown_base(
    topdown_metric: dict,
    subgoals: List[dict],
    sim: Any,
    radius: float,
    *,
    fill_alpha: float = 0.4,
    fill_color: tuple = (255, 0, 255),
    outline_color: tuple = (255, 0, 255),
    point_radius_px: int = 5,
) -> tuple:
    """Colorize the TopDownMap metric and bake in subgoal markers.

    Returns ``(base_bgr, grid_resolution, meters_per_row, meters_per_col)``.
    World -> pixel projection uses ``maps.to_grid`` (same transform the measure
    uses to draw the agent trajectory), so subgoal marks align with the
    trajectory line already drawn on the map.
    """
    from habitat.utils.visualizations import maps as habitat_maps
    from rlinf.envs.habitat.extensions import maps as rlinf_maps

    raw_map = np.asarray(topdown_metric["map"])
    fog_mask = topdown_metric.get("fog_of_war_mask")
    rgb = rlinf_maps.colorize_topdown_map(raw_map, fog_mask)
    base = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    lower_bound, upper_bound = sim.pathfinder.get_bounds()
    meters_per_row = abs(upper_bound[2] - lower_bound[2]) / float(raw_map.shape[0])
    meters_per_col = abs(upper_bound[0] - lower_bound[0]) / float(raw_map.shape[1])

    grid_resolution = (int(raw_map.shape[0]), int(raw_map.shape[1]))
    overlay = base.copy()
    centers: List[tuple] = []
    for item in subgoals:
        pos = item.get("subgoal_position")
        if not pos or len(pos) < 3:
            continue
        row, col = habitat_maps.to_grid(
            float(pos[2]), float(pos[0]), grid_resolution, sim=sim
        )
        centers.append((int(col), int(row)))
        cv2.circle(overlay, (int(col), int(row)), int(point_radius_px), fill_color, -1)

    if centers:
        base = cv2.addWeighted(base, 1.0 - float(fill_alpha), overlay, float(fill_alpha), 0)

    half_w = max(1.0, float(radius) / meters_per_col) if meters_per_col > 0 else 1.0
    half_h = max(1.0, float(radius) / meters_per_row) if meters_per_row > 0 else 1.0
    axes = (int(round(half_w)), int(round(half_h)))
    for center in centers:
        cv2.ellipse(base, center, axes, 0, 0, 360, outline_color, 2, lineType=cv2.LINE_AA)

    return base, grid_resolution, meters_per_row, meters_per_col


def write_topdown_image_from_memory(
    output_path: Path,
    topdown_metric: dict,
    subgoals: List[dict],
    sim: Any,
    radius: float,
    *,
    max_size: int = 0,
    fill_alpha: float = 0.4,
    fill_color: tuple = (255, 0, 255),
    outline_color: tuple = (255, 0, 255),
    point_radius_px: int = 15,
) -> None:
    """Render the final top-down map with all subgoals marked."""
    base, _grid_resolution, _meters_per_row, _meters_per_col = _build_topdown_base(
        topdown_metric,
        subgoals,
        sim,
        radius,
        fill_alpha=fill_alpha,
        fill_color=fill_color,
        outline_color=outline_color,
        point_radius_px=point_radius_px,
    )

    if int(max_size) > 0 and (base.shape[0] > int(max_size) or base.shape[1] > int(max_size)):
        scale = float(int(max_size)) / float(max(base.shape[0], base.shape[1]))
        base = cv2.resize(
            base,
            (int(base.shape[1] * scale), int(base.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), base)
    if not ok:
        raise RuntimeError(f"Failed to write topdown image: {output_path}")
