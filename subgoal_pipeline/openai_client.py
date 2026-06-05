import argparse
import base64
import io
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


def build_system_prompt(min_step_gap: int) -> str:
    return f"""
You are a multimodal indoor-navigation sub-goal selector.

You will be given:
1) One original English navigation instruction.
2) A temporally ordered list of frames from the executed episode. Each frame has a step index.

Your job:
- Do NOT segment, rewrite, paraphrase, or produce sub-instructions.
- Identify the landmark sequence implied by the original instruction.
- Select keyframes as sub-goals for those landmarks.

Landmark rules:
- Prefer explicit landmarks from the instruction.
- If the instruction uses an abstract region such as room, bedroom, bathroom, lobby, hall, hallway, corridor, upstairs, downstairs, or outside, infer a concrete visible proxy landmark from context.
- For entering or exiting a room, the proxy landmark is usually the relevant door or doorway.
- For moving into a hallway/corridor, the proxy landmark is usually the hallway entrance, doorway, or turn point.
- Each sub-goal should use one clear landmark description.

Keyframe rules:
- Do not select the first frame where the landmark becomes barely visible.
- Select a frame where the agent is close enough to the landmark and the landmark is salient.
- Keyframes must be strictly increasing in time.
- Neighboring keyframes should be at least {int(min_step_gap)} steps apart unless there is no reasonable alternative.
- Do not reuse the same step for different sub-goals.
- Avoid tiny initial rotations, startup adjustments, and near-stationary moments.

Final-goal rule:
- The final sub-goal MUST be the episode goal.
- The final sub-goal should correspond to the last/goal frame among the provided frames.
- Mark exactly one sub-goal as is_final_goal=true, and it must be the last item.

Output requirements:
- Output ONLY one JSON object matching the schema.
- Do not include markdown or extra text.
- Do not use placeholder text like "...".
""".strip()


def build_user_text(instruction: str, steps: List[dict]) -> str:
    frame_info = [
        {
            "idx": idx,
            "step": int(step["step"]),
            "rgb_path": str(step["rgb_path"]),
        }
        for idx, step in enumerate(steps)
    ]
    return (
        "Instruction:\n"
        f"{instruction}\n\n"
        "Frames info (JSON, in temporal order):\n"
        f"{json.dumps(frame_info, ensure_ascii=False)}\n\n"
        "You will also receive all frames as images in the same temporal order."
    )


def build_output_schema() -> dict:
    subgoal_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "subgoal_id": {"type": "integer"},
            "landmark": {"type": "string"},
            "landmark_source": {
                "type": "string",
                "enum": ["explicit", "inferred_from_abstract_region", "final_goal"],
            },
            "best_step": {"type": "integer"},
            "is_final_goal": {"type": "boolean"},
            "reason": {"type": "string"},
        },
        "required": [
            "subgoal_id",
            "landmark",
            "landmark_source",
            "best_step",
            "is_final_goal",
            "reason",
        ],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "subgoals": {
                "type": "array",
                "items": subgoal_schema,
            },
        },
        "required": ["subgoals"],
    }


def image_to_b64(image: np.ndarray, image_format: str, jpeg_quality: int) -> Tuple[str, str]:
    pil = Image.fromarray(image)
    buffer = io.BytesIO()
    if image_format == "jpeg":
        pil.save(buffer, format="JPEG", quality=int(jpeg_quality))
        mime = "image/jpeg"
    else:
        pil.save(buffer, format="PNG")
        mime = "image/png"
    return mime, base64.b64encode(buffer.getvalue()).decode("ascii")


def call_openai_for_episode(
    client: Any,
    args: argparse.Namespace,
    instruction: str,
    sampled_steps: List[Any],
    episode_label: str,
) -> Tuple[dict, dict]:
    frame_infos = [
        {
            "step": int(step.step),
            "rgb_path": f"memory_frame_{idx:04d}.{args.image_format}",
        }
        for idx, step in enumerate(sampled_steps)
    ]
    user_content: List[dict] = [
        {
            "type": "input_text",
            "text": build_user_text(instruction=instruction, steps=frame_infos),
        }
    ]

    for step in sampled_steps:
        mime, b64 = image_to_b64(
            image=step.rgb,
            image_format=str(args.image_format),
            jpeg_quality=int(args.jpeg_quality),
        )
        user_content.append({"type": "input_image", "image_url": f"data:{mime};base64,{b64}"})

    print(
        f"  -> OpenAI subgoal select: model={args.model} frames={len(sampled_steps)} "
        f"reasoning_effort={args.reasoning_effort}"
    )
    t0 = time.time()
    resp = client.responses.create(
        model=str(args.model),
        reasoning={"effort": str(args.reasoning_effort)},
        max_output_tokens=int(args.max_output_tokens),
        input=[
            {"role": "system", "content": build_system_prompt(min_step_gap=int(args.min_step_gap))},
            {"role": "user", "content": user_content},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "navigation_subgoals",
                "strict": True,
                "schema": build_output_schema(),
            }
        },
    )
    print(f"  <- OpenAI returned in {time.time() - t0:.1f}s")

    usage = extract_usage_dict(resp)
    text = extract_output_text(resp)
    if not text:
        raise ValueError(f"{episode_label}: empty response from OpenAI.")
    out = json.loads(text)
    if not isinstance(out, dict):
        raise ValueError(f"{episode_label}: OpenAI output is not a JSON object.")
    return out, usage


def extract_output_text(resp: Any) -> str:
    output_text = getattr(resp, "output_text", None)
    if output_text:
        return str(output_text)
    if isinstance(resp, dict):
        output_text = resp.get("output_text")
        if output_text:
            return str(output_text)
        output = resp.get("output") or []
    else:
        output = getattr(resp, "output", None) or []

    chunks: List[str] = []
    for item in output:
        content = item.get("content") if isinstance(item, dict) else getattr(item, "content", None)
        if not content:
            continue
        for part in content:
            if isinstance(part, dict):
                if part.get("type") in {"output_text", "text"} and part.get("text"):
                    chunks.append(str(part["text"]))
            else:
                text = getattr(part, "text", None)
                if text:
                    chunks.append(str(text))
    return "".join(chunks).strip()


def sanitize_model_subgoals(
    raw_subgoals: List[dict],
    valid_steps: set,
    final_step: int,
    min_step_gap: int,
) -> Tuple[List[dict], List[str]]:
    notes: List[str] = []
    candidates: List[dict] = []

    for raw in raw_subgoals:
        if not isinstance(raw, dict):
            notes.append("dropped a non-object subgoal from model output")
            continue
        try:
            best_step = int(raw["best_step"])
        except Exception:
            notes.append(f"dropped subgoal without integer best_step: {raw!r}")
            continue
        if best_step not in valid_steps:
            if bool(raw.get("is_final_goal", False)):
                notes.append(
                    f"forced final sub-goal best_step from unavailable step {best_step} to final step {final_step}"
                )
                best_step = int(final_step)
            else:
                notes.append(f"dropped subgoal with best_step not in sampled frames: {best_step}")
                continue

        candidates.append(
            {
                "subgoal_id": int(raw.get("subgoal_id", len(candidates))),
                "landmark": str(raw.get("landmark") or "").strip() or "landmark",
                "landmark_source": str(raw.get("landmark_source") or "explicit"),
                "best_step": best_step,
                "is_final_goal": bool(raw.get("is_final_goal", False)),
                "reason": str(raw.get("reason") or "").strip(),
            }
        )

    if not candidates:
        candidates.append(
            {
                "subgoal_id": 0,
                "landmark": "episode goal",
                "landmark_source": "final_goal",
                "best_step": int(final_step),
                "is_final_goal": True,
                "reason": "Fallback final sub-goal forced by the script.",
            }
        )
        notes.append("model returned no usable subgoals; created a final-goal-only result")

    candidates = sorted(candidates, key=lambda x: int(x["best_step"]))
    final_candidates = [item for item in candidates if item.get("is_final_goal")]
    if final_candidates:
        final_source: Optional[dict] = final_candidates[-1]
        final_item = dict(final_source)
        if int(final_item["best_step"]) != int(final_step):
            notes.append(
                f"forced final sub-goal best_step from {int(final_item['best_step'])} to final step {int(final_step)}"
            )
        final_item["best_step"] = int(final_step)
        final_item["is_final_goal"] = True
        final_item["landmark_source"] = "final_goal"
        if not final_item.get("landmark"):
            final_item["landmark"] = "episode goal"
    else:
        final_source = None
        final_item = {
            "subgoal_id": len(candidates),
            "landmark": "episode goal",
            "landmark_source": "final_goal",
            "best_step": int(final_step),
            "is_final_goal": True,
            "reason": "Final sub-goal added by the script because the model did not mark one.",
        }
        notes.append("added missing final episode-goal sub-goal")

    non_final = [
        dict(item)
        for item in candidates
        if (final_source is None or item is not final_source) and int(item["best_step"]) < int(final_step)
    ]
    non_final = sorted(non_final, key=lambda x: int(x["best_step"]))

    kept: List[dict] = []
    for item in non_final:
        step = int(item["best_step"])
        if kept and step - int(kept[-1]["best_step"]) < int(min_step_gap):
            prev = kept[-1]
            kept[-1] = item
            notes.append(f"replaced too-close sub-goal step {int(prev['best_step'])} with later step {step}")
            continue
        kept.append(item)

    while kept and int(final_step) - int(kept[-1]["best_step"]) < int(min_step_gap):
        dropped = kept.pop()
        notes.append(f"dropped sub-goal too close to final goal at step {int(dropped['best_step'])}")

    sanitized = kept + [final_item]
    for idx, item in enumerate(sanitized):
        item["subgoal_id"] = idx
        item["is_final_goal"] = idx == len(sanitized) - 1
    return sanitized, notes


def to_plain_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [to_plain_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_plain_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(k): to_plain_jsonable(v) for k, v in value.items()}
    if hasattr(value, "model_dump"):
        return to_plain_jsonable(value.model_dump())
    if hasattr(value, "to_dict"):
        return to_plain_jsonable(value.to_dict())
    if hasattr(value, "__dict__"):
        return {
            str(k): to_plain_jsonable(v)
            for k, v in vars(value).items()
            if not str(k).startswith("_")
        }
    return str(value)


def extract_usage_dict(resp: Any) -> dict:
    usage = getattr(resp, "usage", None)
    if usage is None and isinstance(resp, dict):
        usage = resp.get("usage")
    usage_dict = to_plain_jsonable(usage)
    return usage_dict if isinstance(usage_dict, dict) else {}


def usage_token_counts(usage: dict) -> dict:
    input_tokens = _usage_int(usage, "input_tokens") or _usage_int(usage, "prompt_tokens")
    output_tokens = _usage_int(usage, "output_tokens") or _usage_int(usage, "completion_tokens")
    total_tokens = _usage_int(usage, "total_tokens")
    if not total_tokens and (input_tokens or output_tokens):
        total_tokens = input_tokens + output_tokens
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def _usage_int(usage: dict, key: str) -> int:
    value = usage.get(key)
    if isinstance(value, (int, float)):
        return int(value)
    return 0


def usage_record(
    args: argparse.Namespace,
    out_dir: Path,
    meta: dict,
    token_usage: dict,
    num_frames: int,
) -> dict:
    usage_counts = usage_token_counts(token_usage)
    return {
        "episode_dir": out_dir.name,
        "episode_id": int(meta["episode_id"]),
        "trajectory_id": int(meta["trajectory_id"]),
        "scene_id": meta.get("scene_id"),
        "model": str(args.model),
        "reasoning_effort": str(args.reasoning_effort),
        "num_frames": int(num_frames),
        "input_tokens": usage_counts["input_tokens"],
        "output_tokens": usage_counts["output_tokens"],
        "total_tokens": usage_counts["total_tokens"],
        "usage": token_usage,
        "output_path": str(out_dir / "subgoals_openai.json"),
    }
