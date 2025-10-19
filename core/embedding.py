from __future__ import annotations

import os
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


def run_embedding_selection(txt_path: str, top_n: int = 10) -> str:
    """读取txt段落，按与参考关键词相似度排序，选Top-N写出到 Embedding_<base>.txt"""
    paragraphs = _read_paragraphs(txt_path)
    if not paragraphs:
        # 空输入则直接复制为Embedding文件
        base = os.path.splitext(os.path.basename(txt_path))[0]
        out_path = os.path.join(os.path.dirname(txt_path), f"Embedding_{base}.txt")
        _write_paragraphs([], out_path)
        return out_path

    model = SentenceTransformer("all-MiniLM-L6-v2")
    keywords = [
        # 对齐旧项目关键词集合（含常见变体）
        "flow chemistry continuous flow microflow process development",
        "residence time RT flow rate mL/min mL·min−1 mL·min-1 µL/min uL/min",
        "reactor tubular coil microreactor microchannel packed bed packed tubular ID i.d. inner diameter mm μm",
        "temperature T °C K pressure bar BPR back pressure",
        "conversion yield selectivity product distribution"
    ]
    ref_text = " ".join(keywords)

    # embeddings
    para_vecs = model.encode(paragraphs)
    ref_vec = model.encode(ref_text)

    # 计算相似度
    import numpy as np
    sims = cosine_similarity(np.asarray(para_vecs), np.asarray(ref_vec).reshape(1, -1)).reshape(-1)
    # 选Top-N
    idx_sorted = np.argsort(-sims)[: max(top_n, 1)]
    selected = [paragraphs[i] for i in idx_sorted]

    base = os.path.splitext(os.path.basename(txt_path))[0]
    out_path = os.path.join(os.path.dirname(txt_path), f"Embedding_{base}.txt")
    _write_paragraphs(selected, out_path)
    return out_path


