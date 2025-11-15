from __future__ import annotations

import os
import re
from typing import List

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


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


def _write_paragraphs(paragraphs: List[str], out_path: str) -> None:
    with open(out_path, 'w', encoding='utf-8') as f:
        for p in paragraphs:
            f.write(p + '\n\n')


def clean_chunks(chunks: List[str]) -> List[str]:
    """
    在 Embedding 之前过滤掉无效的文本块。
    规则：
    - 过短块（<10词）
    - 期刊/页眉（Journal of ... (YEAR) VOL:PAGE–PAGE、DOI、版权）
    - 图表/方案题注（Fig/Figure/Scheme/Table N）
    - 板块噪声（Keywords/Graphical Abstract/Supplementary Information/Acknowledgements/Declarations/References）
    - 纯数字（可能为页码）
    """
    cleaned: List[str] = []
    journal_pattern = re.compile(r"Journal of [A-Za-z\s]+ \(\d{4}\) \d+:\d+–\d+", re.IGNORECASE)
    caption_pattern = re.compile(r"^(Fig|Figure|Scheme|Table)\s*\.?\s*\d+", re.IGNORECASE)
    section_pattern = re.compile(
        r"^(Keywords|Graphical Abstract|Supplementary Information|Acknowledgements|Declarations|References)\b",
        re.IGNORECASE,
    )
    doi_pattern = re.compile(r"(^https?://|^doi:|^10\.\d{4,9}/)", re.IGNORECASE)
    copyright_pattern = re.compile(r"©|All rights reserved", re.IGNORECASE)

    for chunk in chunks:
        text = (chunk or "").strip()
        if not text:
            continue
        # 1) 长度阈值（按词数）
        if len(text.split()) < 10:
            continue
        # 2) 纯数字
        if text.isdigit():
            continue
        # 3) 期刊/页眉/DOI/版权
        if journal_pattern.search(text) or doi_pattern.search(text) or copyright_pattern.search(text):
            continue
        # 4) 图表/方案题注
        if caption_pattern.search(text):
            continue
        # 5) 板块噪声
        if section_pattern.search(text):
            continue
        cleaned.append(text)
    return cleaned


def run_embedding_selection(txt_path: str, top_n: int = 10) -> str:
    """读取txt段落，按与参考关键词相似度排序，选Top-N写出到 Embedding_<base>.txt"""
    paragraphs = _read_paragraphs(txt_path)
    if not paragraphs:
        # 空输入则直接复制为Embedding文件
        base = os.path.splitext(os.path.basename(txt_path))[0]
        out_path = os.path.join(os.path.dirname(txt_path), f"Embedding_{base}.txt")
        _write_paragraphs([], out_path)
        return out_path

    # 先进行规则清洗
    paragraphs = clean_chunks(paragraphs)
    if not paragraphs:
        base = os.path.splitext(os.path.basename(txt_path))[0]
        out_path = os.path.join(os.path.dirname(txt_path), f"Embedding_{base}.txt")
        _write_paragraphs([], out_path)
        return out_path

    model = SentenceTransformer("all-MiniLM-L6-v2")

    # 多查询向量（条件/装置/结果），覆盖单位和关键结果词
    queries = {
        "conditions": (
            "flow chemistry reaction conditions parameters flow rate residence time RT "
            "temperature °C K pressure bar BPR back pressure concentration mL/h mL/min µL/min uL/min"
        ),
        "equipment": (
            "reactor setup coil tubular microreactor microchannel packed bed packed tubular "
            "ID i.d. inner diameter mm μm"
        ),
        "outcomes": (
            "optimal yield conversion selectivity productivity mg/h g/h percent % product distribution"
        ),
    }

    # 编码段落与查询
    para_vecs = model.encode(paragraphs)
    ref_vecs = {k: model.encode(v) for k, v in queries.items()}

    # 计算三组相似度，取最大值作为基准分
    import numpy as np
    sims_list = []
    for key in ("conditions", "equipment", "outcomes"):
        v = np.asarray(ref_vecs[key]).reshape(1, -1)
        sims_k = cosine_similarity(np.asarray(para_vecs), v).reshape(-1)
        sims_list.append(sims_k)
    sims_stack = np.vstack(sims_list)  # (3, N)
    base_scores = sims_stack.max(axis=0)  # (N,)

    # 规则加权：命中数值/单位/结果词等给予小幅加权（封顶）
    percent_re = re.compile(r"\b\d{1,3}\s?%\b")
    units_re = re.compile(
        r"(?:\bmL\s*/\s*(?:h|min)\b|\bµL\s*/\s*min\b|\buL\s*/\s*min\b|\bbar\b|\b°C\b|\bmg\s*/\s*h\b)",
        re.IGNORECASE,
    )
    outcomes_re = re.compile(r"\b(yield|conversion|selectivity|productivity)\b", re.IGNORECASE)
    rt_re = re.compile(r"\b(residence time|RT)\b", re.IGNORECASE)
    flow_re = re.compile(r"\bflow rate\b", re.IGNORECASE)
    bpr_re = re.compile(r"\bBPR\b", re.IGNORECASE)

    bonuses = np.zeros_like(base_scores)
    for i, text in enumerate(paragraphs):
        bonus = 0.0
        if percent_re.search(text):
            bonus += 0.12
        if units_re.search(text):
            bonus += 0.08
        if outcomes_re.search(text):
            bonus += 0.08
        if rt_re.search(text):
            bonus += 0.06
        if flow_re.search(text):
            bonus += 0.06
        if bpr_re.search(text):
            bonus += 0.04
        if bonus > 0.25:
            bonus = 0.25
        bonuses[i] = bonus

    final_scores = base_scores + bonuses
    # 选Top-N
    idx_sorted = np.argsort(-final_scores)[: max(top_n, 1)]
    selected = [paragraphs[i] for i in idx_sorted]

    base = os.path.splitext(os.path.basename(txt_path))[0]
    out_path = os.path.join(os.path.dirname(txt_path), f"Embedding_{base}.txt")
    _write_paragraphs(selected, out_path)
    return out_path


