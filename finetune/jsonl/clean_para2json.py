#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clean and validate para2json training data.
Reads finetune/jsonl/para2json.jsonl and writes finetune/jsonl/para2json.cleaned.jsonl

Rules enforced:
- Assistant MUST be ONLY valid JSON (no examples/explanations/fences), starting with '{' and ending with '}'.
- Prohibit ellipses '...'/'…' → set to null.
- Normalize percent numbers: 97% → 97; unit must be '%'.
- Normalize condition keys: temperature, residence_time, flow_rate_total, pressure (lower snake case).
- Allow chemicals without explicit role: keep name, set role to null.
- Reactor: keep only dimension-like values in inner_diameter (e.g., '0.5 mm'); otherwise set inner_diameter to null and keep textual info in reactor.type.
- Strip unknown extraneous keys safely.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List

HERE = os.path.abspath(os.path.dirname(__file__))
SRC = os.path.join(HERE, "para2json.jsonl")
OUT = os.path.join(HERE, "para2json.cleaned.jsonl")


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
                # skip broken line
                continue
    return items


def write_jsonl(path: str, items: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


RE_CODE_FENCE = re.compile(r"```(?:json)?\s*", re.IGNORECASE)
RE_EXAMPLE_INPUT = re.compile(r"(?is)\bExample\s+Input:.*?$")
RE_EXAMPLE_OUTPUT = re.compile(r"(?is)\bExample\s+Output:.*?$")
RE_PERCENT_NUM = re.compile(r'(?<!")(?P<num>\d+(?:\.\d+)?)\s*%')
RE_ELLIPSIS = re.compile(r':\s*("?\.\.\."?|…)(\s*[,}\]])')

COND_KEY_MAP = {
    "Temperature": "temperature",
    "temperature": "temperature",
    "Residence_time": "residence_time",
    "residence_time": "residence_time",
    "Flow_rate_total": "flow_rate_total",
    "flow_rate_total": "flow_rate_total",
    "Pressure": "pressure",
    "pressure": "pressure",
}
ALLOWED_COND_TYPES = {"temperature", "residence_time", "flow_rate_total", "pressure"}


def extract_json_block(s: str) -> str:
    s = s.strip()
    s = RE_CODE_FENCE.sub("", s)
    # cut example appendices
    s = RE_EXAMPLE_INPUT.sub("", s).strip()
    s = RE_EXAMPLE_OUTPUT.sub("", s).strip()
    # keep largest {...} block
    l, r = s.find("{"), s.rfind("}")
    if l != -1 and r != -1 and r > l:
        s = s[l : r + 1]
    return s


def looks_mm(val: str) -> bool:
    if not isinstance(val, str):
        return False
    return bool(re.search(r"\b\d+\.?\d*\s*mm\b", val.lower()))


def sanitize_json_text(txt: str) -> str:
    s = extract_json_block(txt)
    # convert 97% → 97 (unquoted)
    s = RE_PERCENT_NUM.sub(r"\g<num>", s)
    # unifying condition types (by textual replacement)
    for k, v in COND_KEY_MAP.items():
        s = s.replace(f'"{k}"', f'"{v}"')
    # ellipsis → null
    s = RE_ELLIPSIS.sub(r": null\2", s)
    return s


def coerce_numeric(val: Any) -> Any:
    if isinstance(val, (int, float)) or val is None:
        return val
    if isinstance(val, str):
        t = val.strip()
        t = t.replace("%", "")
        try:
            return float(t) if "." in t else int(t)
        except Exception:
            return None
    return None


def ensure_schema(obj: Dict[str, Any]) -> Dict[str, Any]:
    # minimal skeleton
    rs = obj.get("reaction_summary") or {}
    out: Dict[str, Any] = {
        "reaction_summary": {
            "reaction_type": rs.get("reaction_type"),
            "reactants": [],
            "products": [],
            "conditions": [],
            "reactor": {"type": None, "inner_diameter": None},
            "metrics": {"conversion": None, "yield": None, "selectivity": None, "unit": "%"},
        }
    }
    # reactants
    reactants = rs.get("reactants") or []
    fixed_reactants = []
    if isinstance(reactants, list):
        for it in reactants:
            if isinstance(it, dict):
                name = it.get("name")
                role = it.get("role", None)
                fixed_reactants.append({"name": name, "role": role})
            elif isinstance(it, str):
                fixed_reactants.append({"name": it, "role": None})
    out["reaction_summary"]["reactants"] = fixed_reactants
    # products
    products = rs.get("products") or []
    fixed_products = []
    if isinstance(products, list):
        for it in products:
            if isinstance(it, dict):
                fixed_products.append(
                    {
                        "name": it.get("name"),
                        "yield_optimal": coerce_numeric(it.get("yield_optimal")),
                        "unit": "%",
                    }
                )
            elif isinstance(it, str):
                fixed_products.append({"name": it, "yield_optimal": None, "unit": "%"})
    out["reaction_summary"]["products"] = fixed_products
    # conditions
    conditions = rs.get("conditions") or []
    fixed_conditions = []
    if isinstance(conditions, list):
        for it in conditions:
            if not isinstance(it, dict):
                continue
            t = it.get("type")
            v = it.get("value")
            if isinstance(t, str):
                t = COND_KEY_MAP.get(t, t)
            if t in ALLOWED_COND_TYPES:
                fixed_conditions.append({"type": t, "value": v if (v is None or isinstance(v, str)) else str(v)})
    out["reaction_summary"]["conditions"] = fixed_conditions
    # reactor
    reactor = rs.get("reactor") or {}
    r_type = reactor.get("type")
    r_id = reactor.get("inner_diameter")
    # inner_diameter only keeps mm-like values
    if isinstance(r_id, str) and not looks_mm(r_id):
        r_id = None
    out["reaction_summary"]["reactor"] = {"type": r_type, "inner_diameter": r_id}
    # metrics
    metrics = rs.get("metrics") or {}
    conv = coerce_numeric(metrics.get("conversion"))
    yld = coerce_numeric(metrics.get("yield"))
    sel = coerce_numeric(metrics.get("selectivity"))
    out["reaction_summary"]["metrics"] = {
        "conversion": conv,
        "yield": yld,
        "selectivity": sel,
        "unit": "%",
    }
    # reaction_type passthrough/null
    out["reaction_summary"]["reaction_type"] = rs.get("reaction_type")
    return out


def clean_item(ex: Dict[str, Any]) -> Dict[str, Any] | None:
    msgs = ex.get("messages") or []
    if not isinstance(msgs, list) or not msgs:
        return None
    # find assistant
    sys_idx = next((i for i, m in enumerate(msgs) if m.get("role") == "system"), None)
    usr_idx = next((i for i, m in enumerate(msgs) if m.get("role") == "user"), None)
    asst_idx = next((i for i, m in enumerate(msgs) if m.get("role") == "assistant"), None)
    if asst_idx is None or usr_idx is None:
        return None
    asst = msgs[asst_idx].get("content") or ""
    cleaned_txt = sanitize_json_text(str(asst))
    try:
        obj = json.loads(cleaned_txt)
    except Exception:
        # second-chance hard strip: keep only braces
        cleaned_txt2 = extract_json_block(cleaned_txt)
        try:
            obj = json.loads(cleaned_txt2)
        except Exception:
            return None
    obj2 = ensure_schema(obj if isinstance(obj, dict) else {})
    # rebuild messages: keep original system/user, replace assistant with normalized JSON
    new_msgs: List[Dict[str, Any]] = []
    if sys_idx is not None:
        new_msgs.append({"role": "system", "content": msgs[sys_idx].get("content", "")})
    new_msgs.append({"role": "user", "content": msgs[usr_idx].get("content", "")})
    new_msgs.append({"role": "assistant", "content": json.dumps(obj2, ensure_ascii=False)})
    return {"messages": new_msgs}


def main() -> None:
    if not os.path.exists(SRC):
        print(f"Source not found: {SRC}")
        return
    data = read_jsonl(SRC)
    cleaned: List[Dict[str, Any]] = []
    total = len(data)
    ok = 0
    for ex in data:
        res = clean_item(ex)
        if res is not None:
            cleaned.append(res)
            ok += 1
    write_jsonl(OUT, cleaned)
    print(f"Total: {total}, Cleaned: {ok}, Output: {OUT}")


if __name__ == "__main__":
    main()









