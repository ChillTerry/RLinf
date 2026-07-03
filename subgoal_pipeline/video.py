import subprocess
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image, ImageDraw, ImageFont

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


def resolve_font_path(font_path: str = "") -> str:
    candidates = []
    if font_path:
        candidates.append(Path(font_path))
    candidates.extend(
        [
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
            Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError("No usable font found. Pass --video_font_path explicitly.")


def draw_center_highlight(
    overlay: Image.Image,
    text: str,
    font: ImageFont.FreeTypeFont,
    box_color: tuple,
    text_fill: tuple,
    stroke_fill: tuple,
    padding_x: int,
    padding_y: int,
    line_spacing: int,
) -> None:
    draw = ImageDraw.Draw(overlay)
    max_width = int(overlay.width * 0.82)
    words = str(text).split()
    lines: List[str] = []
    cur = ""
    for word in words or [str(text)]:
        trial = word if not cur else f"{cur} {word}"
        bbox = draw.textbbox((0, 0), trial, font=font, stroke_width=2)
        if bbox[2] - bbox[0] <= max_width or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)

    line_boxes = [draw.textbbox((0, 0), line, font=font, stroke_width=2) for line in lines]
    widths = [box[2] - box[0] for box in line_boxes]
    heights = [box[3] - box[1] for box in line_boxes]
    text_width = max(widths) if widths else 0
    text_height = sum(heights) + line_spacing * max(0, len(lines) - 1)
    box_w = text_width + padding_x * 2
    box_h = text_height + padding_y * 2
    x0 = (overlay.width - box_w) // 2
    y0 = (overlay.height - box_h) // 2
    draw.rounded_rectangle((x0, y0, x0 + box_w, y0 + box_h), radius=8, fill=box_color)

    y = y0 + padding_y
    for line, box, height in zip(lines, line_boxes, heights):
        line_w = box[2] - box[0]
        x = x0 + (box_w - line_w) // 2
        draw.text((x, y), line, font=font, fill=text_fill, stroke_width=2, stroke_fill=stroke_fill)
        y += height + line_spacing


def write_subgoal_video_from_memory(
    output_path: Path,
    steps: List[MemoryStep],
    subgoals: List[dict],
    fps: int,
    font_path: str,
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

            frame = Image.fromarray(step_info.rgb).convert("RGBA")
            overlay = Image.new("RGBA", frame.size, (0, 0, 0, 0))
            if highlight_remaining > 0 and active_landmark:
                center_font = ImageFont.truetype(font_path, size=max(28, height // 16))
                draw_center_highlight(
                    overlay,
                    active_landmark,
                    center_font,
                    box_color=(0, 0, 0, 150),
                    text_fill=(255, 40, 40, 255),
                    stroke_fill=(0, 0, 0, 255),
                    padding_x=22,
                    padding_y=16,
                    line_spacing=8,
                )
            vis = Image.alpha_composite(frame, overlay).convert("RGB")
            writer.write_rgb(np.asarray(vis))

            if highlight_remaining > 0:
                highlight_remaining -= 1
                if highlight_remaining == 0:
                    active_landmark = ""
    finally:
        writer.close()

