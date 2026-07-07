import subprocess
from pathlib import Path
from typing import Dict, List

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
) -> None:
    if not steps:
        return
    first = steps[0].rgb
    height, width = int(first.shape[0]), int(first.shape[1])
    writer = VideoWriter(output_path=output_path, width=width, height=height, fps=int(fps))
    highlight_by_step: Dict[int, str] = {
        int(item["best_step"]): str(item.get("landmark") or "")
        for item in subgoals
    }
    highlight_frames = max(1, int(highlight_frames))
    highlight_remaining = 0
    active_landmark = ""

    try:
        for step_info in steps:
            if int(step_info.step) in highlight_by_step:
                active_landmark = highlight_by_step[int(step_info.step)]
                highlight_remaining = highlight_frames

            frame = np.asarray(step_info.rgb, dtype=np.uint8)
            if highlight_remaining > 0 and active_landmark:
                frame = draw_center_highlight(frame, active_landmark)
            writer.write_rgb(frame)

            if highlight_remaining > 0:
                highlight_remaining -= 1
                if highlight_remaining == 0:
                    active_landmark = ""
    finally:
        writer.close()
