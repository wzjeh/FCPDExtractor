#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute offline stats for para2json cleaned dataset:
- JSON parse rate of assistant outputs
- Field coverage: reactants/products/conditions/reactor/metrics

Inputs:
  - finetune/jsonl/para2json.cleaned.jsonl (preferred)
  - fallback: finetune/jsonl/para2json.jsonl (best-effort parse)

Outputs:
  - Console summary
  - finetune/jsonl/para2json.cleaned.stats.json
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

HERE = os.path.abspath(os.path.dirname(__file__))
PREFERRED = os.path.join(HERE, "para2json.mix_7_3.jsonl")
FALLBACK = os.path.join(HERE, "para2json.jsonl")
REPORT = os.path.join(HERE, "para2json.mix_7_3.stats.json")

ALLOWED_COND_TYPES = {"temperature", "residence_time", "flow_rate_total", "pressure"}


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return items
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                # ignore bad lines
                continue
    return items


def get_assistant_text(msgs: List[Dict[str, Any]]) -> str | None:
    for m in msgs or []:
        if m.get("role") == "assistant":
            return m.get("content")
    return None


def parse_assistant_json(text: str) -> Dict[str, Any] | None:
    try:
        return json.loads(text)
    except Exception:
        return None


def has_reactant_name(obj: Dict[str, Any]) -> bool:
    rs = obj.get("reaction_summary", {}) or {}
    for r in rs.get("reactants", []) or []:
        if isinstance(r, dict) and r.get("name"):
            return True
    return False


def has_product_signal(obj: Dict[str, Any]) -> bool:
    rs = obj.get("reaction_summary", {}) or {}
    for p in rs.get("products", []) or []:
        if not isinstance(p, dict):
            continue
        if p.get("name") or isinstance(p.get("yield_optimal"), (int, float)):
            return True
    return False


def has_condition_value(obj: Dict[str, Any]) -> Tuple[bool, Dict[str, int]]:
    """Return any-condition-present flag and per-type counts for allowed types."""
    rs = obj.get("reaction_summary", {}) or {}
    conds = rs.get("conditions", []) or []
    per_type: Dict[str, int] = {t: 0 for t in ALLOWED_COND_TYPES}
    any_ok = False
    for c in conds:
        if not isinstance(c, dict):
            continue
        t = c.get("type")
        v = c.get("value")
        if t in ALLOWED_COND_TYPES and v:
            any_ok = True
            per_type[t] = per_type.get(t, 0) + 1
    return any_ok, per_type


def has_reactor_info(obj: Dict[str, Any]) -> bool:
    rs = obj.get("reaction_summary", {}) or {}
    reactor = rs.get("reactor") or {}
    if not isinstance(reactor, dict):
        return False
    return bool(reactor.get("type") or reactor.get("inner_diameter"))


def has_metrics_numeric(obj: Dict[str, Any]) -> bool:
    rs = obj.get("reaction_summary", {}) or {}
    met = rs.get("metrics") or {}
    if not isinstance(met, dict):
        return False
    for k in ("conversion", "yield", "selectivity"):
        if isinstance(met.get(k), (int, float)):
            return True
    return False


def compute_stats(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(items)
    parsed_ok = 0
    cov_reactant = 0
    cov_product = 0
    cov_condition = 0
    cov_reactor = 0
    cov_metrics = 0
    per_type_counts: Dict[str, int] = {t: 0 for t in ALLOWED_COND_TYPES}

    for ex in items:
        msgs = ex.get("messages")
        if not isinstance(msgs, list):
            continue
        atext = get_assistant_text(msgs)
        if not atext:
            continue
        jobj = parse_assistant_json(atext)
        if not isinstance(jobj, dict):
            continue
        parsed_ok += 1

        if has_reactant_name(jobj):
            cov_reactant += 1
        if has_product_signal(jobj):
            cov_product += 1
        cond_ok, per_type = has_condition_value(jobj)
        if cond_ok:
            cov_condition += 1
        for k in ALLOWED_COND_TYPES:
            per_type_counts[k] += per_type.get(k, 0)
        if has_reactor_info(jobj):
            cov_reactor += 1
        if has_metrics_numeric(jobj):
            cov_metrics += 1

    def pct(n: int) -> float:
        return (n / total * 100.0) if total else 0.0

    summary = {
        "total": total,
        "parsed_ok": parsed_ok,
        "parse_rate_pct": (parsed_ok / total * 100.0) if total else 0.0,
        "coverage": {
            "reactant_name_cnt": cov_reactant,
            "reactant_name_pct": pct(cov_reactant),
            "product_signal_cnt": cov_product,
            "product_signal_pct": pct(cov_product),
            "any_condition_cnt": cov_condition,
            "any_condition_pct": pct(cov_condition),
            "reactor_info_cnt": cov_reactor,
            "reactor_info_pct": pct(cov_reactor),
            "metrics_numeric_cnt": cov_metrics,
            "metrics_numeric_pct": pct(cov_metrics),
        },
        "conditions_type_counts": per_type_counts,
    }
    return summary


def main() -> None:
    src = PREFERRED if os.path.exists(PREFERRED) else FALLBACK
    if not os.path.exists(src):
        print("No dataset found. Expected one of:")
        print(" -", PREFERRED)
        print(" -", FALLBACK)
        return
    items = read_jsonl(src)
    stats = compute_stats(items)
    print("=== para2json cleaned stats ===")
    print("Source:", src)
    print("Total:", stats["total"])
    print("Parsed OK:", stats["parsed_ok"], f"({stats['parse_rate_pct']:.2f}%)")
    cov = stats["coverage"]
    print("Coverage:")
    print("  - reactant name:", cov["reactant_name_cnt"], f"({cov['reactant_name_pct']:.2f}%)")
    print("  - product signal:", cov["product_signal_cnt"], f"({cov['product_signal_pct']:.2f}%)")
    print("  - any condition:", cov["any_condition_cnt"], f"({cov['any_condition_pct']:.2f}%)")
    print("  - reactor info  :", cov["reactor_info_cnt"], f"({cov['reactor_info_pct']:.2f}%)")
    print("  - metrics numeric:", cov["metrics_numeric_cnt"], f"({cov['metrics_numeric_pct']:.2f}%)")
    print("Condition type counts:", stats["conditions_type_counts"])
    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print("Report written:", REPORT)


if __name__ == "__main__":
    main()








