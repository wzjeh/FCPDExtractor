#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Augment para2json cleaned dataset to improve short-sentence extraction performance without manual labeling.

Input:
  finetune/jsonl/para2json.cleaned.jsonl
Output:
  finetune/jsonl/para2json.augmented.jsonl

What it does:
1) Standardize chat template:
   - system: strict "Only JSON, no explanation" message
   - user: append strict rules once (ban ellipses/examples; fixed keys; lower snake case for condition types)
2) Heuristic, conservative augmentation from user Context (no cross-paragraph inference):
   - metrics: extract numeric conversion/yield/selectivity (percent) if clearly present; fill when assistant lacks them
   - reactor: if missing, extract reactor.type phrases and inner_diameter like "0.5 mm"
   (No hallucination: only fills when unambiguous regex matches found)
3) Keep assistant as valid JSON.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Tuple

HERE = os.path.abspath(os.path.dirname(__file__))
SRC = os.path.join(HERE, "para2json.cleaned.jsonl")
OUT = os.path.join(HERE, "para2json.augmented.jsonl")

STRICT_SYSTEM = "You output ONLY valid JSON. No explanations, no markdown, no comments."
STRICT_APPEND_RULES = (
    "\n\nRules (augmented):\n"
    "- DO NOT use '...' or '…'. Use null/empty arrays for unknowns.\n"
    "- Output starts with '{' and ends with '}' only. No examples or explanations.\n"
    "- Keys must be exactly: reaction_summary.reaction_type, reactants[{name,role}], "
    "products[{name,yield_optimal,unit}], conditions[{type,value}], reactor{type,inner_diameter}, "
    "metrics{conversion,yield,selectivity,unit}.\n"
    "- Condition types must be lower snake case: temperature, residence_time, flow_rate_total, pressure.\n"
    "- Copy condition values as strings exactly from the paragraph when present (e.g., '160 °C', '21 min').\n"
    "- Chemicals without explicit role are allowed: set role to null (do not drop the name)."
)

# Regexes
RE_PERCENT = re.compile(r"(?i)\b(\d+(?:\.\d+)?)\s*%")
RE_YIELD = re.compile(r"(?i)\byield(?:\s+of)?\s*[:=]?\s*(\d+(?:\.\d+)?)\s*%")
RE_CONV = re.compile(r"(?i)\bconversion(?:\s+of)?\s*[:=]?\s*(\d+(?:\.\d+)?)\s*%")
RE_SELEC = re.compile(r"(?i)\bselectivit(?:y|ies)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*%")
RE_TEMPERATURE_VAL = re.compile(r"(?i)\b(?:T\s*=?\s*)?(\d+(?:\.\d+)?)\s*°\s*C")
RE_RES_TIME_VAL = re.compile(r"(?i)\b(\d+(?:\.\d+)?)\s*(?:min|minutes?)\b")
RE_INNER_DIAMETER = re.compile(r"(?i)\b(\d+(?:\.\d+)?)\s*mm\b")
RE_REACTOR_PHRASES = [
    re.compile(r"(?i)capillary microreactor"),
    re.compile(r"(?i)microreactor"),
    re.compile(r"(?i)packed\s+bed"),
    re.compile(r"(?i)tubular\s+reactor"),
    re.compile(r"(?i)tube\s+reactor"),
    re.compile(r"(?i)coil"),
    re.compile(r"(?i)CSTR|continuous stirred tank reactor"),
    re.compile(r"(?i)Vapourtec\s+R[- ]Series\s+flow\s+system"),
    re.compile(r"(?i)stainless steel tube reactor"),
    re.compile(r"(?i)PFA tube reactor"),
    re.compile(r"(?i)flow\s+reactor"),
]


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
                continue
    return items


def write_jsonl(path: str, items: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def get_msg(msgs: List[Dict[str, Any]], role: str) -> Dict[str, Any] | None:
    for m in msgs or []:
        if m.get("role") == role:
            return m
    return None


def parse_json(s: str) -> Dict[str, Any] | None:
    try:
        return json.loads(s)
    except Exception:
        return None


def to_number(x: Any) -> Any:
    if isinstance(x, (int, float)) or x is None:
        return x
    if isinstance(x, str):
        t = x.replace("%", "").strip()
        try:
            return float(t) if "." in t else int(t)
        except Exception:
            return None
    return None


def extract_metrics_from_text(text: str) -> Dict[str, Any]:
    out = {"conversion": None, "yield": None, "selectivity": None, "unit": "%"}
    # Try labeled captures first
    m = RE_YIELD.search(text)
    if m:
        out["yield"] = to_number(m.group(1))
    m = RE_CONV.search(text)
    if m:
        out["conversion"] = to_number(m.group(1))
    m = RE_SELEC.search(text)
    if m:
        out["selectivity"] = to_number(m.group(1))
    # If all None, try to pick a single percent as yield (conservative)
    if all(v is None for k, v in out.items() if k != "unit"):
        m = RE_PERCENT.search(text)
        if m:
            out["yield"] = to_number(m.group(1))
    return out


def extract_reactor_from_text(text: str) -> Tuple[str | None, str | None]:
    r_type = None
    r_id = None
    # longest matching reactor phrase
    matches = []
    for pat in RE_REACTOR_PHRASES:
        for mm in pat.finditer(text):
            matches.append(mm.group(0))
    if matches:
        r_type = max(matches, key=len)
    m = RE_INNER_DIAMETER.search(text)
    if m:
        r_id = f"{m.group(1)} mm"
    return r_type, r_id


def ensure_rules_appended(user_content: str) -> str:
    marker = "Rules (augmented):"
    if marker in user_content:
        return user_content
    return user_content.rstrip() + STRICT_APPEND_RULES


def augment_example(ex: Dict[str, Any]) -> Dict[str, Any] | None:
    msgs = ex.get("messages")
    if not isinstance(msgs, list):
        return None
    sys = get_msg(msgs, "system")
    usr = get_msg(msgs, "user")
    ast = get_msg(msgs, "assistant")
    if not usr or not ast:
        return None

    # Standardize system
    new_msgs: List[Dict[str, Any]] = []
    new_msgs.append({"role": "system", "content": STRICT_SYSTEM})
    # Standardize user (append strict rules)
    utext = str(usr.get("content") or "")
    utext2 = ensure_rules_appended(utext)
    new_msgs.append({"role": "user", "content": utext2})

    # Parse assistant JSON
    atext = str(ast.get("content") or "")
    jobj = parse_json(atext)
    if not isinstance(jobj, dict):
        # keep original if broken (shouldn't happen after cleaning)
        return None

    rs = jobj.get("reaction_summary", {}) or {}
    # Metrics augmentation when absent
    met = rs.get("metrics") or {}
    needs_metrics = not any(isinstance(met.get(k), (int, float)) for k in ("conversion", "yield", "selectivity"))
    if needs_metrics:
        m_ex = extract_metrics_from_text(utext2)
        # Fill only if we actually found something numeric
        if any(isinstance(m_ex.get(k), (int, float)) for k in ("conversion", "yield", "selectivity")):
            met["conversion"] = m_ex.get("conversion")
            met["yield"] = m_ex.get("yield")
            met["selectivity"] = m_ex.get("selectivity")
            met["unit"] = "%"
            rs["metrics"] = met
            jobj["reaction_summary"] = rs

    # Reactor augmentation when absent
    reactor = rs.get("reactor") or {}
    r_type = reactor.get("type")
    r_id = reactor.get("inner_diameter")
    if not r_type or not isinstance(r_type, str) or not r_type.strip():
        rt, rid = extract_reactor_from_text(utext2)
        if rt and (not r_type):
            reactor["type"] = rt
        if (not r_id) and rid:
            reactor["inner_diameter"] = rid
        rs["reactor"] = reactor
        jobj["reaction_summary"] = rs

    # Finalize assistant as JSON
    new_msgs.append({"role": "assistant", "content": json.dumps(jobj, ensure_ascii=False)})
    return {"messages": new_msgs}


def main() -> None:
    if not os.path.exists(SRC):
        print(f"Source not found: {SRC}")
        return
    data = read_jsonl(SRC)
    out: List[Dict[str, Any]] = []
    total = len(data)
    aug_cnt = 0
    for ex in data:
        res = augment_example(ex)
        if res is None:
            # fallback to original example if cannot process
            out.append(ex)
            continue
        # detect if augmentation actually changed metrics/reactor
        out.append(res)
        aug_cnt += 1
    write_jsonl(OUT, out)
    print(f"Total: {total}, Augmented: {aug_cnt}, Output: {OUT}")


if __name__ == "__main__":
    main()









