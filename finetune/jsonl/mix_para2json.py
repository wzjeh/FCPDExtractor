#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create a mixed para2json training set with cleaned:augmented ≈ 7:3 (no duplicate pairs).

By default expects:
  - finetune/jsonl/para2json.cleaned.jsonl
  - finetune/jsonl/para2json.augmented.jsonl
Outputs:
  - finetune/jsonl/para2json.mix_7_3.jsonl

Strategy:
  - Assumes augmented was generated 1:1 from cleaned (same order/size).
  - Randomly pick ~30% indices to take augmented; others take cleaned.
  - Ensures each original pair contributes exactly one example (no duplicates).
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Dict, List

HERE = os.path.abspath(os.path.dirname(__file__))
DEFAULT_CLEANED = os.path.join(HERE, "para2json.cleaned.jsonl")
DEFAULT_AUG = os.path.join(HERE, "para2json.augmented.jsonl")
DEFAULT_OUT = os.path.join(HERE, "para2json.mix_7_3.jsonl")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                # skip broken lines
                continue
    return items


def write_jsonl(path: str, items: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def make_mix(cleaned: List[Dict[str, Any]], augmented: List[Dict[str, Any]],
             cleaned_ratio: float, seed: int = 42) -> List[Dict[str, Any]]:
    assert len(cleaned) == len(augmented), "cleaned 与 augmented 数量不一致，无法 1:1 配对混合"
    n = len(cleaned)
    rng = random.Random(seed)
    # 选择使用 augmented 的索引集合（约 1 - cleaned_ratio）
    aug_count = int(round(n * (1.0 - cleaned_ratio)))
    aug_indices = set(rng.sample(range(n), aug_count))
    out: List[Dict[str, Any]] = []
    for i in range(n):
        if i in aug_indices:
            out.append(augmented[i])
        else:
            out.append(cleaned[i])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Mix para2json cleaned and augmented sets (7:3 by default).")
    parser.add_argument("--cleaned", type=str, default=DEFAULT_CLEANED, help="Path to cleaned jsonl")
    parser.add_argument("--augmented", type=str, default=DEFAULT_AUG, help="Path to augmented jsonl")
    parser.add_argument("--output", type=str, default=DEFAULT_OUT, help="Output mixed jsonl")
    parser.add_argument("--cleaned_ratio", type=float, default=0.7, help="Portion from cleaned (0~1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic split")
    args = parser.parse_args()

    if not os.path.exists(args.cleaned):
        print(f"not found: {args.cleaned}")
        return
    if not os.path.exists(args.augmented):
        print(f"not found: {args.augmented}")
        return

    cleaned = read_jsonl(args.cleaned)
    augmented = read_jsonl(args.augmented)
    mixed = make_mix(cleaned, augmented, args.cleaned_ratio, seed=args.seed)
    write_jsonl(args.output, mixed)
    print(f"Cleaned: {len(cleaned)}  Augmented: {len(augmented)}")
    print(f"Mixed: {len(mixed)}  cleaned_ratio={args.cleaned_ratio}  seed={args.seed}")
    print(f"→ {args.output}")


if __name__ == "__main__":
    main()









