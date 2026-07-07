#!/usr/bin/env python3
"""Extract English (en-US, en-IN) entries from RxR reachable + gt_reachable json.gz files.

Usage:
    python extract_english.py <reachable.json.gz> <gt_reachable.json.gz>

Writes `<stem>_english.json.gz` next to each input.
"""
import gzip
import json
import sys
from pathlib import Path

EN_LANGS = {"en-US", "en-IN"}


def load(path):
    with gzip.open(path, "rt") as f:
        return json.load(f)


def dump(obj, path):
    with gzip.open(path, "wt") as f:
        json.dump(obj, f, ensure_ascii=False)


def extract(reachable_path, gt_path):
    # 1. reachable: filter episodes by instruction.language
    rd = load(reachable_path)
    eps = rd["episodes"]
    ep_lang = {
        str(e["episode_id"]): e["instruction"]["language"]
        for e in eps
        if e.get("instruction", {}).get("language")
    }
    en_eps = [e for e in eps if ep_lang.get(str(e["episode_id"])) in EN_LANGS]

    out_reachable = Path(reachable_path).with_name(
        Path(reachable_path).stem.replace(".json", "_english.json.gz")
    )
    rd_en = dict(rd)
    rd_en["episodes"] = en_eps
    dump(rd_en, out_reachable)
    print(
        f"{Path(reachable_path).name}: {len(eps)} -> {len(en_eps)} (en-US+en-IN) "
        f"-> {out_reachable.name}"
    )

    # 2. gt_reachable: keep keys whose episode_id is English
    gt = load(gt_path)
    gt_en = {k: v for k, v in gt.items() if ep_lang.get(k) in EN_LANGS}

    out_gt = Path(gt_path).with_name(
        Path(gt_path).stem.replace(".json", "_english.json.gz")
    )
    dump(gt_en, out_gt)
    print(
        f"{Path(gt_path).name}: {len(gt)} -> {len(gt_en)} (en-US+en-IN) "
        f"-> {out_gt.name}"
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    extract(sys.argv[1], sys.argv[2])
