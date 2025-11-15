from __future__ import annotations

import os
import re
import json
import argparse
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

import sys
from pathlib import Path

from gpt4all import GPT4All
from openai import OpenAI

# 确保可以从项目根目录导入 core/*
try:
    repo_root = Path(__file__).resolve().parents[2]
    if repo_root.exists():
        sys.path.insert(0, str(repo_root))
except Exception:
    pass

from core.text_utils import extract_text_from_pdf, write_text, split_paragraphs
from core.embedding import run_embedding_selection
from core.models.qwen_llm import QwenLLM


def _read_paragraphs(txt_path: str) -> List[str]:
    with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    segs: List[str] = []
    cur: List[str] = []
    for line in lines:
        if line.strip():
            cur.append(line.strip())
        else:
            if cur:
                segs.append(' '.join(cur))
                cur = []
    if cur:
        segs.append(' '.join(cur))
    return segs


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _create_prompt(system: str, user: str, context: str = "") -> str:
    parts: List[str] = []
    if system:
        parts.append(system)
    if context:
        parts.append(f"Context\n{context}")
    parts.append(f"Task\n{user}")
    return "\n\n".join(parts)


def ensure_pdf_to_txt(pdf_path: str, out_txt_path: str, *, resume: bool = True) -> str:
    if resume and os.path.exists(out_txt_path) and os.path.getsize(out_txt_path) > 0:
        return out_txt_path
    text = extract_text_from_pdf(pdf_path)
    paragraphs = split_paragraphs(text, min_len=60)
    _ensure_dir(os.path.dirname(out_txt_path))
    write_text("\n\n".join(paragraphs), out_txt_path)
    return out_txt_path


KEYWORDS = [
    "flow chemistry", "continuous flow", "residence time", "flow rate", "mL/min", "µL/min", "ul/min",
    "reactor", "tubular", "coil", "microreactor", "inner diameter", "i.d.", "mm", "μm",
    "temperature", "°c", "selectivity", "conversion", "yield", "bpr", "bar", "back pressure", "min", "pressure"
]


def load_local_stage_model(models_root: str, model_filename: str) -> GPT4All:
    # GPT4All 接口会在 model_path 下查找 model_filename
    return GPT4All(model_filename, model_path=models_root, allow_download=False)


def filter_paragraphs_with_local_model(paragraphs: List[str], model: GPT4All) -> List[str]:
    system_prompt = (
        "You are an expert assistant for scientific literature mining. "
        "Classify paragraphs as Yes/No based on whether they contain concrete experimental details."
    )
    user_question = (
        "Does the paragraph contain experimental details about flow-chemistry/process development? "
        "Answer strictly with 'Yes' or 'No'."
    )
    kept: List[str] = []
    for p in paragraphs:
        low = p.lower()
        if any(k in low for k in KEYWORDS):
            kept.append(p)
            continue
        prompt = _create_prompt(system_prompt, user_question, p[:1200])
        try:
            resp = (model.generate(prompt=prompt, max_tokens=5, temp=0.0) or "").strip().lower()
        except Exception:
            resp = ""
        if resp.startswith("yes"):
            kept.append(p)
    return kept


def abstract_paragraphs_with_local_model(paragraphs: List[str], model: GPT4All) -> List[str]:
    system_prompt = "You are an expert assistant for scientific literature mining. Return concise, faithful summaries."
    user_prompt = (
        "Summarize the paragraph focusing on flow-chemistry process development. "
        "Highlight: reaction type, key reactants/solvent/catalyst, products, reactor details (type/ID), "
        "critical conditions (flow rate(s), residence time, temperature, pressure), and outcomes (conversion/yield/selectivity). "
        "Be concise, faithful to text, no speculation."
    )
    abstracts: List[str] = []
    for p in paragraphs:
        prompt = _create_prompt(system_prompt, user_prompt, p[:2000])
        try:
            text = model.generate(prompt=prompt, max_tokens=280, temp=0.0) or ""
        except Exception:
            text = ""
        abstracts.append((text.strip() or p[:400]).strip())
    return abstracts


SENT_SPLIT_RE = re.compile(r'(?<=[。！？；;.!?])\s+')


def split_sentences(text: str) -> List[str]:
    if not text:
        return []
    parts = SENT_SPLIT_RE.split(text.strip())
    return [s.strip() for s in parts if s.strip()]


JSON_SYSTEM_PROMPT = "You output ONLY valid JSON. No explanations, no markdown, no comments."

# 与示例一致的任务描述（保持字段、规则与示例）
JSON_TASK_TEMPLATE = (
    "Only use the provided paragraph; do not infer across other paragraphs.\n"
    "STRICT EXTRACTION POLICY:\n"
    "- Extract ONLY entities that are explicitly and unambiguously named in THIS paragraph.\n"
    "- DO NOT GUESS or INVENT any chemical, role, condition, reactor, product or metrics.\n"
    "- If role for a chemical is not explicitly stated near its name, OMIT that chemical entirely (prefer an empty list over guessing).\n"
    "- SKIP generic placeholders (e.g., monomer(s), reactant(s), feedstock(s), mixture, solution, linear chains, arm precursor, star core, reaction mixture).\n"
    "- Deduplicate case-insensitively; prefer singular names; keep explicit abbreviations as shown in text (e.g., BA, ACMO, DMF, ABVN, DoPAT, HDDA).\n"
    "- If conflicting info exists, choose null and DO NOT resolve by speculation.\n"
    "If a field is not explicitly stated, use null. Use original units when present; otherwise normalize: "
    "temperature in °C, residence_time in min, flow_rate in mL/min, inner_diameter in mm.\n"
    "Output ONLY the following JSON object (no extra text):\n"
    "{ \"reaction_summary\": {  \"reaction_type\":\"...\",   \"reactants\":[{\"name\":\"...\",\"role\":\"reactant|catalyst|solvent\"}],   "
    "\"products\":[{\"name\":\"...\",\"yield_optimal\":95,\"unit\":\"%\"}],   \"conditions\":[    "
    "{\"type\":\"temperature\",\"value\":\"...\"},    {\"type\":\"residence_time\",\"value\":\"...\"},    "
    "{\"type\":\"flow_rate_reactant_A\",\"value\":\"...\"},    {\"type\":\"flow_rate_total\",\"value\":\"...\"},    "
    "{\"type\":\"pressure\",\"value\":\"...\"}  ],   \"reactor\":{\"type\":\"...\",\"inner_diameter\":\"...\"},   "
    "\"metrics\":{\"conversion\":...,\"yield\":...,\"selectivity\":...,\"unit\":\"%\"}}}\n"
    "Example input: \"Flow rate 0.1 mL/min, T=80 °C in a 0.5 mm coil; yield 82%.\"\n"
    "Example output: { \"reaction_summary\": {  \"reaction_type\": null, \"reactants\": [],  "
    "\"products\": [{\"name\": null, \"yield_optimal\": 82, \"unit\": \"%\"}],  \"conditions\": [ "
    "{\"type\":\"temperature\",\"value\":\"80 °C\"}, {\"type\":\"flow_rate_total\",\"value\":\"0.1 mL/min\"} ],  "
    "\"reactor\": {\"type\":\"coil\", \"inner_diameter\":\"0.5 mm\"},  \"metrics\": {\"conversion\": null, \"yield\": 82, "
    "\"selectivity\": null, \"unit\": \"%\"}}}\n"
    "Rules:\n"
    "- For CONDITIONS and METRICS: choose the OPTIMAL set (highest yield/conversion).\n"
    "- For reaction_type, reactants, products, reactor: use the most informative/complete data (not necessarily from the optimal condition).\n"
    "- If multiple conditions appear, output only ONE optimal condition set.\n"
    "- Use null for unknown fields."
)

GENERIC_REJECT = {
    "monomer", "monomers", "reactant", "reactants", "feedstock", "feedstocks",
    "mixture", "solution", "linear chains", "linear chain", "arm precursor",
    "star core", "reaction mixture", "unreacted monomer", "initiators",
    "synthesized linear polymers", "polymer chains", "arms", "core"
}

SYNONYM_MAP = {
    "ba": "BA",
    "acmo": "ACMO",
    "dmf": "DMF",
    "abvn": "ABVN",
    "dopat": "DoPAT",
    "hdda": "HDDA",
    "l-pacmo": "l-PACMO",
}

ALLOWED_ROLES = {"reactant", "catalyst", "solvent", "initiator"}

def _normalize_chem_name(name: str) -> str:
    s = (name or "").strip()
    if not s:
        return s
    low = s.lower()
    if low in SYNONYM_MAP:
        return SYNONYM_MAP[low]
    # 简单去尾部's'
    if low.endswith('s') and len(low) > 3 and low[:-1] not in SYNONYM_MAP:
        s = s[:-1]
    return s

def _is_generic(name: str) -> bool:
    if not name:
        return True
    low = name.strip().lower()
    return low in GENERIC_REJECT

def sanitize_reaction_summary(obj: Dict[str, Any], *, strict_no_infer: bool = True) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        return obj
    rs = obj.get("reaction_summary")
    if not isinstance(rs, dict):
        return obj

    # Reactants: 过滤通用词、规范名称、大小写无关去重；若 role 不在允许集合且严格模式，则丢弃该条
    reactants = []
    seen_names = set()
    for r in (rs.get("reactants") or []):
        if not isinstance(r, dict):
            continue
        name = _normalize_chem_name(r.get("name") or "")
        role = r.get("role")
        if not name or _is_generic(name):
            continue
        if strict_no_infer and (role is None or str(role).strip().lower() not in ALLOWED_ROLES):
            # 严格模式下，角色不明确不收
            continue
        role_norm = str(role).strip().lower() if role else None
        key = (name.lower(), role_norm)
        if key in seen_names:
            continue
        seen_names.add(key)
        # 保留原大小写名称，但角色统一小写
        item = {"name": name}
        if role_norm:
            item["role"] = role_norm
        reactants.append(item)
    rs["reactants"] = reactants

    # Products: 允许通用名，但做去重（按名称），保留最高yield_optimal
    products_best: Dict[str, Dict[str, Any]] = {}
    for p in (rs.get("products") or []):
        if not isinstance(p, dict):
            continue
        name = p.get("name")
        key = (name or "").strip()
        cur = products_best.get(key, {"name": name, "yield_optimal": None, "unit": p.get("unit", "%") or "%"})
        y = p.get("yield_optimal")
        try:
            yv = float(y) if y is not None else None
        except Exception:
            yv = None
        try:
            cv = float(cur.get("yield_optimal")) if cur.get("yield_optimal") is not None else None
        except Exception:
            cv = None
        if cv is None or (yv is not None and yv > cv):
            cur["yield_optimal"] = y
        products_best[key] = cur
    rs["products"] = list(products_best.values())

    # Metrics单位统一'%'
    metrics = rs.get("metrics") or {}
    if isinstance(metrics, dict):
        metrics["unit"] = "%"
        rs["metrics"] = metrics

    obj["reaction_summary"] = rs
    return obj


def _extract_json_obj(text: str) -> Dict[str, Any] | None:
    # 尝试直接解析；失败则提取首个最外层花括号片段
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r'\{[\s\S]*\}', text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def _build_user_content_for_sentence(sentence: str) -> str:
    return f"Context\n{sentence}\n\nTask\n{JSON_TASK_TEMPLATE}"


def _collect_processed_sentences(jsonl_path: str) -> Set[str]:
    done: Set[str] = set()
    if not os.path.exists(jsonl_path):
        return done
    with open(jsonl_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                msgs = obj.get("messages", [])
                for m in msgs:
                    if m.get("role") == "user":
                        content = m.get("content", "")
                        # 提取 Context 部分
                        if content.startswith("Context\n"):
                            # 拆出 Context 与 Task 之间文本
                            parts = content.split("\n\nTask", 1)
                            if parts:
                                s = parts[0].replace("Context\n", "", 1).strip()
                                if s:
                                    done.add(s)
                        break
            except Exception:
                continue
    return done


def qwen_extract_json_for_sentence(
    sentence: str,
    llm: QwenLLM,
    *,
    temperature: float = 0.0,
    top_p: float | None = 0.6,
    max_retries: int = 2,
    cooldown: float = 1.2,
) -> str | None:
    # 返回规范化后的 JSON 字符串（用于 messages.assistant.content）
    system = JSON_SYSTEM_PROMPT
    user = _build_user_content_for_sentence(sentence)
    prompt = _create_prompt(system, user, context="")
    attempt = 0
    while attempt <= max_retries:
        try:
            raw = llm.generate(prompt, temperature=temperature, top_p=top_p, max_tokens=512) or ""
        except Exception:
            raw = ""
        obj = _extract_json_obj(raw)
        if obj is not None:
            try:
                # 只保留最小化JSON，确保是有效对象
                normalized = json.dumps(obj, ensure_ascii=False, separators=(',', ':'))
                return normalized
            except Exception:
                pass
        attempt += 1
        time.sleep(cooldown * (2 ** attempt))
    return None


def process_one_pdf(
    pdf_path: str,
    *,
    output_root: str,
    models_root: str,
    filter_model_name: str,
    abstract_model_name: str,
    top_n: int,
    resume: bool,
    do_abstract: bool = False,
) -> Dict[str, str]:
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    paper_dir = os.path.join(output_root, base)
    txt_path = os.path.join(paper_dir, f"{base}.txt")

    # 1) PDF -> txt
    ensure_pdf_to_txt(pdf_path, txt_path, resume=resume)

    # 2) Embedding + TopN
    embedding_txt = run_embedding_selection(txt_path, top_n=top_n)

    # 3) Local filter
    paras = _read_paragraphs(embedding_txt)
    filter_model = load_local_stage_model(models_root, filter_model_name)
    kept = filter_paragraphs_with_local_model(paras, filter_model)
    filtered_path = os.path.join(paper_dir, f"Embedding_{base}_Filtered.txt")
    write_text("\n\n".join(kept), filtered_path)

    abstract_path = ""
    if do_abstract:
        # 4) Local abstract（可选）
        abstract_model = load_local_stage_model(models_root, abstract_model_name)
        abstracts = abstract_paragraphs_with_local_model(kept, abstract_model)
        abstract_path = os.path.join(paper_dir, f"Embedding_{base}_Filtered_Abstract.txt")
        write_text("\n\n".join(abstracts), abstract_path)

    return {
        "txt": txt_path,
        "embedding": embedding_txt,
        "filtered": filtered_path,
        "abstract": abstract_path,
    }


def build_jsonl_from_text(
    text_path: str,
    *,
    jsonl_output: str,
    qwen_model_name: str,
    qwen_api_key_env: str,
    concurrency: int,
    resume: bool,
    temperature: float,
    top_p: float | None,
    merge_output_path: str | None = None,
    granularity: str = "sentence",  # "sentence" | "paragraph"
    min_chars: int = 0,
) -> Tuple[int, int]:
    # 收集已完成句子用于断点续跑
    done_sentences = _collect_processed_sentences(jsonl_output) if resume else set()

    # 读入抽象段落并切分句子
    paragraphs = _read_paragraphs(text_path)
    items: List[str] = []
    if granularity == "paragraph":
        for p in paragraphs:
            s = (p or "").strip()
            if not s:
                continue
            if min_chars and len(s) < min_chars:
                continue
            if not resume or s not in done_sentences:
                items.append(s)
    else:
        for p in paragraphs:
            ss = split_sentences(p)
            for s in ss:
                s = (s or "").strip()
                if not s:
                    continue
                if min_chars and len(s) < min_chars:
                    continue
                if not resume or s not in done_sentences:
                    items.append(s)

    if not items:
        return (0, 0)

    llm = QwenLLM(api_key_env_var=qwen_api_key_env, model_name=qwen_model_name)

    created, skipped = 0, 0
    merged_objs: List[Dict[str, Any]] = []
    _ensure_dir(os.path.dirname(jsonl_output))
    with open(jsonl_output, 'a', encoding='utf-8') as fout:
        # 并发提取
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
            future_to_idx = {
                ex.submit(qwen_extract_json_for_sentence, s, llm, temperature=temperature, top_p=top_p): (idx, s)
                for idx, s in enumerate(items)
            }
            # 结果按完成顺序写入（不强制全局顺序）
            for fut in as_completed(future_to_idx):
                idx, sent = future_to_idx[fut]
                try:
                    normalized_json = fut.result()
                except Exception:
                    normalized_json = None
                if not normalized_json:
                    skipped += 1
                    continue
                # 收集合并材料
                try:
                    obj_full = json.loads(normalized_json)
                    if isinstance(obj_full, dict):
                        # 严格清洗，移除通用词、去重、规范角色与单位
                        obj_full = sanitize_reaction_summary(obj_full, strict_no_infer=True)
                        normalized_json = json.dumps(obj_full, ensure_ascii=False, separators=(',', ':'))
                        merged_objs.append(obj_full)
                except Exception:
                    # 若解析失败，跳过清洗
                    pass
                # messages 三段式
                messages = [
                    {"role": "system", "content": JSON_SYSTEM_PROMPT},
                    {"role": "user", "content": _build_user_content_for_sentence(sent)},
                    {"role": "assistant", "content": normalized_json},
                ]
                line_obj = {"messages": messages}
                fout.write(json.dumps(line_obj, ensure_ascii=False) + "\n")
                created += 1

    # 可选：写出合并后的整篇JSON
    if merge_output_path and merged_objs:
        def best_metrics_key(rs: Dict[str, Any]) -> Tuple[float, float]:
            m = (rs or {}).get("metrics") or {}
            y = m.get("yield")
            c = m.get("conversion")
            yv = float(y) if isinstance(y, (int, float)) else -1.0
            cv = float(c) if isinstance(c, (int, float)) else -1.0
            return (yv, cv)

        # 收集所有 reaction_summary
        summaries: List[Dict[str, Any]] = []
        for o in merged_objs:
            rs = o.get("reaction_summary")
            if isinstance(rs, dict):
                summaries.append(rs)

        agg: Dict[str, Any] = {"reaction_summary": {
            "reaction_type": None,
            "reactants": [],
            "products": [],
            "conditions": [],
            "reactor": {"type": None, "inner_diameter": None},
            "metrics": {"conversion": None, "yield": None, "selectivity": None, "unit": "%"},
        }}
        if summaries:
            # reaction_type: 选最长的非空字符串
            types = [s.get("reaction_type") for s in summaries if isinstance(s.get("reaction_type"), str) and s.get("reaction_type").strip()]
            if types:
                agg["reaction_summary"]["reaction_type"] = max(types, key=len)

            # reactants: 去重(name.lower(), role)
            seen_r: Set[Tuple[str, str]] = set()
            reactants_out: List[Dict[str, Any]] = []
            for s in summaries:
                for r in (s.get("reactants") or []):
                    name = r.get("name")
                    role = r.get("role")
                    if not name or not role:
                        continue
                    key = (str(name).strip().lower(), str(role).strip())
                    if key in seen_r:
                        continue
                    seen_r.add(key)
                    reactants_out.append({"name": name, "role": role})
            agg["reaction_summary"]["reactants"] = reactants_out

            # products: 按名称聚合，保留最高yield_optimal
            prod_best: Dict[str, Dict[str, Any]] = {}
            null_prod_best: float | None = None
            for s in summaries:
                for p in (s.get("products") or []):
                    name = p.get("name")
                    y = p.get("yield_optimal")
                    if name:
                        key = str(name).strip()
                        cur = prod_best.get(key, {"name": name, "yield_optimal": None, "unit": "%"})
                        try:
                            yv = float(y) if y is not None else None
                        except Exception:
                            yv = None
                        try:
                            cv = float(cur.get("yield_optimal")) if cur.get("yield_optimal") is not None else None
                        except Exception:
                            cv = None
                        if cv is None or (yv is not None and yv > cv):
                            cur["yield_optimal"] = y
                        prod_best[key] = cur
                    else:
                        try:
                            yv = float(y) if y is not None else None
                        except Exception:
                            yv = None
                        if yv is not None and (null_prod_best is None or yv > null_prod_best):
                            null_prod_best = yv
            products_out = list(prod_best.values())
            if not products_out and null_prod_best is not None:
                products_out = [{"name": None, "yield_optimal": null_prod_best, "unit": "%"}]
            agg["reaction_summary"]["products"] = products_out

            # 选最佳metrics来源的conditions与reactor
            best_rs = None
            if summaries:
                best_rs = max(summaries, key=best_metrics_key)
            if best_rs:
                # conditions
                conds = best_rs.get("conditions") or []
                if isinstance(conds, list):
                    agg["reaction_summary"]["conditions"] = conds
                # reactor
                reactor = best_rs.get("reactor") or {}
                rtype = reactor.get("type")
                rid = reactor.get("inner_diameter")
                agg["reaction_summary"]["reactor"] = {"type": rtype, "inner_diameter": rid}

            # metrics: 取全局最大
            best_yield = None
            best_conv = None
            best_sel = None
            for s in summaries:
                m = s.get("metrics") or {}
                for key_name, cur_best in (("yield", "yield"), ("conversion", "conversion"), ("selectivity", "selectivity")):
                    val = m.get(key_name)
                    try:
                        v = float(val) if val is not None else None
                    except Exception:
                        v = None
                    if key_name == "yield":
                        if v is not None and (best_yield is None or v > best_yield):
                            best_yield = v
                    elif key_name == "conversion":
                        if v is not None and (best_conv is None or v > best_conv):
                            best_conv = v
                    elif key_name == "selectivity":
                        if v is not None and (best_sel is None or v > best_sel):
                            best_sel = v
            agg["reaction_summary"]["metrics"]["yield"] = best_yield
            agg["reaction_summary"]["metrics"]["conversion"] = best_conv
            agg["reaction_summary"]["metrics"]["selectivity"] = best_sel

        _ensure_dir(os.path.dirname(merge_output_path))
        with open(merge_output_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(agg, ensure_ascii=False, separators=(',', ':')))

    return (created, skipped)


def load_config(path: str) -> Dict[str, Any]:
    # 只读，不修改现有配置
    import yaml
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def discover_pdfs(input_dir: str, limit: int | None) -> List[str]:
    pdfs: List[str] = []
    for root, _, files in os.walk(input_dir):
        for fn in files:
            if fn.lower().endswith(".pdf"):
                pdfs.append(os.path.join(root, fn))
    pdfs.sort()
    if limit:
        pdfs = pdfs[:max(0, limit)]
    return pdfs


class QwenDirect:
    def __init__(self, api_key_env_var: str, model_name: str, base_url: str) -> None:
        api_key = os.getenv(api_key_env_var)
        if not api_key:
            raise ValueError(f"Environment variable {api_key_env_var} not set.")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    def generate(self, prompt: str, **kwargs: Any) -> str:
        temperature = kwargs.pop("temperature", kwargs.get("temp", 0.0))
        top_p = kwargs.pop("top_p", None)
        max_tokens = kwargs.get("max_tokens", 512)
        req = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if top_p is not None:
            req["top_p"] = top_p
        try:
            resp = self.client.chat.completions.create(**req)
            if resp.choices and len(resp.choices) > 0:
                return (resp.choices[0].message.content or "").strip()
            return ""
        except Exception as e:
            print(f"⚠️ Qwen API错误: {e}")
            return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build JSONL from finetune/papers via local 5-step to Abstract, then Qwen per-sentence JSON.")
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--input_dir', type=str, default='finetune/papers')
    parser.add_argument('--output', type=str, default=None, help='输出 JSONL 文件路径')
    parser.add_argument('--topn', type=int, default=10)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--concurrency', type=int, default=3)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--use_abstract', action='store_true', help='使用抽象文本（默认关闭，直接用过滤文本句子）')
    parser.add_argument('--qwen_model', type=str, default=None)
    parser.add_argument('--qwen_api_key_env', type=str, default=None)
    parser.add_argument('--qwen_temperature', type=float, default=0.0)
    parser.add_argument('--qwen_top_p', type=float, default=0.6)
    parser.add_argument('--qwen_base_url', type=str, default=None, help='可选：覆盖Qwen base_url，例如 https://dashscope.aliyuncs.com/compatible-mode/v1')
    parser.add_argument('--write_merged', action='store_true', help='为每篇论文输出合并后的整篇JSON')
    parser.add_argument('--granularity', type=str, default='sentence', choices=['sentence','paragraph'], help='Qwen提取粒度：句子或段落')
    parser.add_argument('--min_chars', type=int, default=0, help='丢弃短于该字符数的上下文')
    args = parser.parse_args()

    cfg = load_config(args.config)
    # 路径与本地模型
    paths = cfg.get('paths', {})
    default_papers_dir = paths.get('papers_dir', 'data/papers')
    default_output_dir = 'finetune/jsonl'
    input_dir = args.input_dir or default_papers_dir
    os.makedirs(default_output_dir, exist_ok=True)
    out_jsonl = args.output or os.path.join(default_output_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_train_data.jsonl")

    local_cfg = cfg.get('local_model', {})
    models_root = local_cfg.get('path', 'models/')
    filter_model_name = local_cfg.get('filter')
    abstract_model_name = local_cfg.get('abstract')
    if not filter_model_name or not abstract_model_name:
        raise ValueError("config.yaml 缺少 local_model.filter 或 local_model.abstract 配置。")

    # Qwen
    qwen_cfg = cfg.get('qwen_api', {})
    qwen_model_name = args.qwen_model or qwen_cfg.get('model_name', 'qwen-plus')
    qwen_api_key_env = args.qwen_api_key_env or qwen_cfg.get('api_key_env_var', 'QWEN_API_KEY')

    pdfs = discover_pdfs(input_dir, args.limit)
    if not pdfs:
        print("未发现PDF。")
        return

    # 输出根目录放在 data/local/<paper> 下（不改既有命名）
    output_root = os.path.join('data', 'local')
    _ensure_dir(output_root)

    # 抑制 tokenizers 并行警告（仅进程内）
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    print(f"共发现 {len(pdfs)} 篇PDF，TopN={args.topn}，并发={args.concurrency}，resume={args.resume}")
    total_created, total_skipped = 0, 0
    for i, pdf in enumerate(pdfs, 1):
        print(f"[{i}/{len(pdfs)}] 处理: {os.path.basename(pdf)}")
        paths_out = process_one_pdf(
            pdf,
            output_root=output_root,
            models_root=models_root,
            filter_model_name=filter_model_name,
            abstract_model_name=abstract_model_name,
            top_n=args.topn,
            resume=args.resume,
            do_abstract=args.use_abstract,
        )
        src_path = paths_out["abstract"] if (args.use_abstract and paths_out.get("abstract")) else paths_out["filtered"]
        # 选择Qwen客户端：若提供 base_url，则直接走OpenAI兼容端；否则走项目内封装
        if args.qwen_base_url:
            llm = QwenDirect(api_key_env_var=qwen_api_key_env, model_name=qwen_model_name, base_url=args.qwen_base_url)
        else:
            llm = QwenLLM(api_key_env_var=qwen_api_key_env, model_name=qwen_model_name)

        # 执行构建
        merge_path = None
        if args.write_merged:
            merge_path = os.path.join(os.path.dirname(src_path), "Merged_Overall.json")
        created, skipped = build_jsonl_from_text(
            src_path,
            jsonl_output=out_jsonl,
            qwen_model_name=qwen_model_name,
            qwen_api_key_env=qwen_api_key_env,
            concurrency=max(1, args.concurrency),
            resume=args.resume,
            temperature=float(args.qwen_temperature),
            top_p=float(args.qwen_top_p) if args.qwen_top_p is not None else None,
            merge_output_path=merge_path,
            granularity=args.granularity,
            min_chars=int(args.min_chars or 0),
        )
        print(f"  → JSONL新增: {created} 行，跳过: {skipped} 行")
        total_created += created
        total_skipped += skipped

    print(f"完成。生成/追加 JSONL: {out_jsonl}，新增 {total_created} 行，跳过 {total_skipped} 行。")


if __name__ == "__main__":
    main()


