"""Plot GT action length and instruction length distributions for failed episodes.

Reads per-episode eval metrics from ``episode_{id}.json`` files (filename is
the episode id, see ``rlinf/envs/habitat/habitat_env.py``), tags each as
success (``success == 1``) or failure (``success == 0``), then looks up its
GT action length (``len(actions)`` from the ``*_gt_reachable.json.gz``
mapping) and instruction word count (``len(instruction_text.split())`` from
the ``*_reachable.json.gz`` episode file). Produces two stacked histograms
showing where failures fall within the full evaluated distribution.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .common import load_json

_EPISODE_RE = re.compile(r"^episode_(.+)\.json$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=Path(
            "logs/20260705-12:22:40-habitat_rxr_grpo_uninavid/metrics/eval"
        ),
    )
    parser.add_argument(
        "--gt-file",
        type=Path,
        default=Path(
            "VLN-CE/datasets/rxr/val_unseen/val_unseen_guide_gt_reachable_english.json.gz"
        ),
    )
    parser.add_argument(
        "--episodes-file",
        type=Path,
        default=Path(
            "VLN-CE/datasets/rxr/val_unseen/val_unseen_guide_reachable_english.json.gz"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to --eval-dir.",
    )
    parser.add_argument(
        "--bin-width",
        type=int,
        default=20,
        help="Histogram bin width (bin edges at 0, bin_width, 2*bin_width, ...).",
    )
    return parser.parse_args()


def _load_eval(eval_dir: Path) -> Dict[str, dict]:
    """Return {episode_id: metrics} for every episode_*.json in eval_dir."""
    out: Dict[str, dict] = {}
    for p in sorted(eval_dir.glob("episode_*.json")):
        m = _EPISODE_RE.match(p.name)
        if not m:
            continue
        with open(p) as f:
            out[m.group(1)] = json.load(f)
    return out


def _gt_lengths(gt_file: Path) -> Dict[str, int]:
    data = load_json(gt_file)  # {episode_id(str): {actions: [...]}}
    return {str(eid): int(len(v["actions"])) for eid, v in data.items()}


def _instr_lengths(episodes_file: Path) -> Dict[str, int]:
    data = load_json(episodes_file)  # {"episodes": [...]}
    out: Dict[str, int] = {}
    for ep in data.get("episodes", []):
        eid = str(ep["episode_id"])
        text = ep.get("instruction", {}).get("instruction_text", "")
        out[eid] = len(text.split())
    return out


def _split_by_success(
    eval_metrics: Dict[str, dict],
    gt_len: Dict[str, int],
    instr_len: Dict[str, int],
) -> Tuple[List[int], List[int], List[int], List[int], int]:
    """Bucket GT length and instruction length into success/failure lists."""
    gt_succ: List[int] = []
    gt_fail: List[int] = []
    ins_succ: List[int] = []
    ins_fail: List[int] = []
    missing = 0
    for eid, m in eval_metrics.items():
        if eid not in gt_len or eid not in instr_len:
            missing += 1
            continue
        is_success = float(m.get("success", 0)) == 1.0
        g = gt_len[eid]
        i = instr_len[eid]
        if is_success:
            gt_succ.append(g)
            ins_succ.append(i)
        else:
            gt_fail.append(g)
            ins_fail.append(i)
    return gt_succ, gt_fail, ins_succ, ins_fail, missing


def _plot(
    succ: List[int],
    fail: List[int],
    bin_width: int,
    xlabel: str,
    title: str,
    out_path: Path,
) -> None:
    total = len(succ) + len(fail)
    # Bin edges at 0, bin_width, 2*bin_width, ... up to cover the max value.
    max_val = max(succ + fail) if (succ or fail) else 0
    edges = np.arange(0, max_val + bin_width, bin_width)
    succ_counts, _ = np.histogram(succ, bins=edges)
    fail_counts, _ = np.histogram(fail, bins=edges)
    bin_centers = (edges[:-1] + edges[1:]) / 2
    with np.errstate(invalid="ignore", divide="ignore"):
        success_rate = np.where(
            succ_counts + fail_counts > 0,
            succ_counts / (succ_counts + fail_counts),
            np.nan,
        )
    fig, ax = plt.subplots(figsize=(8, 5))
    # Stacked: failure on the bottom (red), success on top (blue, labeled as
    # the full-dataset distribution). The total bar height = full dataset count.
    ax.hist(
        [fail, succ],
        bins=edges,
        stacked=True,
        label=[f"failure (n={len(fail)})", f"full dataset (n={total})"],
        color=["#c44e52", "#4c72b0"],
        edgecolor="white",
        linewidth=0.3,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("episode count")
    ax.set_title(title)

    # Right axis: per-bin success rate.
    ax2 = ax.twinx()
    ax2.plot(
        bin_centers,
        success_rate * 100,
        color="#2ca02c",
        marker="o",
        linewidth=1.5,
        label="success rate (per bin)",
    )
    ax2.set_ylabel("success rate (%)", color="#2ca02c")
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis="y", labelcolor="#2ca02c")
    # Combined legend (bars + line).
    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles + handles2, labels + labels2, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir or args.eval_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_metrics = _load_eval(args.eval_dir)
    gt_len = _gt_lengths(args.gt_file)
    instr_len = _instr_lengths(args.episodes_file)

    gt_succ, gt_fail, ins_succ, ins_fail, missing = _split_by_success(
        eval_metrics, gt_len, instr_len
    )

    print(
        f"eval={len(eval_metrics)} success={len(gt_succ)} "
        f"failure={len(gt_fail)} missing_in_dataset={missing}"
    )

    _plot(
        gt_succ,
        gt_fail,
        args.bin_width,
        "GT action length (len(actions))",
        "GT action length: failure vs full dataset",
        out_dir / "failure_dist_gt_len.png",
    )
    _plot(
        ins_succ,
        ins_fail,
        args.bin_width,
        "Instruction word count",
        "Instruction length: failure vs full dataset",
        out_dir / "failure_dist_instr_len.png",
    )
    print(f"saved -> {out_dir / 'failure_dist_gt_len.png'}")
    print(f"saved -> {out_dir / 'failure_dist_instr_len.png'}")


if __name__ == "__main__":
    main()
