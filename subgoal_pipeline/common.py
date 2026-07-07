import gzip
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


ACTION_ID_TO_NAME = {
    0: "STOP",
    1: "MOVE_FORWARD",
    2: "TURN_LEFT",
    3: "TURN_RIGHT",
}


def is_gzip_path(path: Path) -> bool:
    if str(path).endswith(".gz"):
        return True
    if not path.exists():
        return False
    with path.open("rb") as f:
        return f.read(2) == b"\x1f\x8b"


def load_json(path: Path) -> dict:
    opener = gzip.open if is_gzip_path(path) else open
    with opener(path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise RuntimeError(f"{path} is not a JSON object.")
    return data


def write_json(path: Path, data: dict, pretty: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "wt", encoding="utf-8") as f:
        if pretty:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
        else:
            json.dump(data, f, separators=(",", ":"), ensure_ascii=False)


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def to_float_list(values: Any) -> List[float]:
    return [float(v) for v in (values or [])]


def instruction_text(episode: dict) -> str:
    instruction = episode.get("instruction")
    if isinstance(instruction, dict):
        return str(instruction.get("instruction_text") or instruction.get("instruction") or "")
    return str(instruction or "")


def episode_language(episode: dict) -> Optional[str]:
    """Return the instruction language tag if present (RxR), else None (R2R)."""
    instruction = episode.get("instruction")
    if isinstance(instruction, dict):
        language = instruction.get("language")
        return str(language) if language is not None else None
    return None


def detect_dataset_type(train_data: dict) -> str:
    """Detect ``'rxr'`` vs ``'r2r'`` from episode structure.

    RxR episodes carry an ``instruction.language`` field (e.g. ``'en-US'``,
    ``'hi-IN'``); R2R episodes do not. Scans episodes until the first language
    tag is found.
    """
    for episode in train_data.get("episodes", []) or []:
        if episode_language(episode) is not None:
            return "rxr"
    return "r2r"


def is_english_instruction(episode: dict) -> bool:
    """True if the episode instruction language is an English variant."""
    language = episode_language(episode)
    return language is not None and language.lower().startswith("en")


def extract_goal_payload(episode: Any) -> Tuple[Optional[List[float]], List[dict]]:
    goals = getattr(episode, "goals", None) or []
    goal_position: Optional[List[float]] = None
    goals_payload: List[dict] = []

    for goal in goals:
        position = getattr(goal, "position", None)
        if position is None and isinstance(goal, dict):
            position = goal.get("position")
        if position is None:
            continue

        pos = to_float_list(position)
        if goal_position is None:
            goal_position = pos

        goal_item: Dict[str, Any] = {"position": pos}
        radius = getattr(goal, "radius", None)
        if radius is None and isinstance(goal, dict):
            radius = goal.get("radius")
        if radius is not None:
            goal_item["radius"] = float(radius)
        goals_payload.append(goal_item)

    return goal_position, goals_payload


def extract_info_payload(episode: Any) -> dict:
    info = getattr(episode, "info", None) or {}
    payload: Dict[str, Any] = {}
    if isinstance(info, dict) and info.get("geodesic_distance") is not None:
        payload["geodesic_distance"] = float(info["geodesic_distance"])
    return payload


def gzip_dataset_for_habitat(source_path: Path, cache_dir: Path) -> Path:
    source_path = source_path.resolve()
    digest = hashlib.sha1(str(source_path).encode("utf-8")).hexdigest()[:10]
    gzip_path = cache_dir / f"{source_path.stem}_{digest}.json.gz"

    cache_dir.mkdir(parents=True, exist_ok=True)
    if gzip_path.exists() and gzip_path.stat().st_mtime >= source_path.stat().st_mtime:
        print(f"Using cached gzip dataset for Habitat: {gzip_path}")
        return gzip_path

    print(f"Habitat R2R loader expects gzip; compressing {source_path} -> {gzip_path}")
    tmp_path = gzip_path.with_suffix(gzip_path.suffix + ".tmp")
    with source_path.open("rb") as src, gzip.open(tmp_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    tmp_path.replace(gzip_path)
    return gzip_path


def prepare_habitat_data_path(data_path: str, split: str, out_dir: str) -> str:
    resolved_path = Path(str(data_path).format(split=split)).expanduser()
    cache_dir = (Path(out_dir).expanduser() / "_habitat_dataset_cache").resolve()

    if resolved_path.exists():
        if is_gzip_path(resolved_path):
            return str(resolved_path)
        return str(gzip_dataset_for_habitat(resolved_path, cache_dir))

    if str(resolved_path).endswith(".gz"):
        plain_path = Path(str(resolved_path)[:-3])
        if plain_path.exists():
            if is_gzip_path(plain_path):
                return str(plain_path)
            return str(gzip_dataset_for_habitat(plain_path, cache_dir))

    raise FileNotFoundError(
        f"Dataset data_path does not exist: {resolved_path}. "
        "Pass a .json.gz file or a plain .json file that can be gzipped automatically."
    )

