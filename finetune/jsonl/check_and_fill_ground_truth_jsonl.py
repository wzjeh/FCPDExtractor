from __future__ import annotations

import os
import re
import argparse
from typing import List, Dict, Any, Set
import sys
from pathlib import Path

# 确保可以从项目根目录导入 finetune/* 与 core/*
try:
    repo_root = Path(__file__).resolve().parents[2]
    if repo_root.exists():
        sys.path.insert(0, str(repo_root))
except Exception:
    pass

from finetune.jsonl.build_from_papers import (
    load_config,
    process_one_pdf,
    build_jsonl_from_text,
    _ensure_dir,
    QwenDirect,  # 可选：支持自定义 base_url
)
from core.models.qwen_llm import QwenLLM


def _discover_ids_with_json(ground_truth_dir: str) -> List[str]:
    ids: List[str] = []
    for fn in os.listdir(ground_truth_dir):
        if fn.endswith("_reaction_annotated.json"):
            m = re.match(r"^(\d+)_", fn)
            if m:
                ids.append(m.group(1))
    ids.sort(key=lambda x: int(x))
    return ids


def _existing_jsonl_ids(ground_truth_dir: str) -> Set[str]:
    have: Set[str] = set()
    for fn in os.listdir(ground_truth_dir):
        if fn.endswith(".jsonl"):
            name = os.path.splitext(fn)[0]
            if name.isdigit():
                have.add(name)
    return have


def main() -> None:
    parser = argparse.ArgumentParser(
        description="检查 finetune/ground_truth 下是否每个标注 json 都有对应 {id}.jsonl，缺失则重新提取并输出到同目录。"
    )
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--ground_truth_dir", type=str, default="finetune/ground_truth")
    parser.add_argument("--papers_dir", type=str, default="finetune/papers")
    parser.add_argument("--topn", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--use_abstract", action="store_true")
    parser.add_argument("--granularity", type=str, default="sentence", choices=["sentence", "paragraph"])
    parser.add_argument("--min_chars", type=int, default=0)
    parser.add_argument("--only_ids", type=str, default=None, help="仅校验并补齐这些编号（逗号分隔）")
    parser.add_argument("--qwen_model", type=str, default=None)
    parser.add_argument("--qwen_api_key_env", type=str, default=None)
    parser.add_argument("--qwen_temperature", type=float, default=0.0)
    parser.add_argument("--qwen_top_p", type=float, default=0.6)
    parser.add_argument("--qwen_base_url", type=str, default=None, help="可选：覆盖Qwen base_url")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # 本地阶段模型
    local_cfg: Dict[str, Any] = cfg.get("local_model", {}) or {}
    models_root = local_cfg.get("path", "models/")
    filter_model_name = local_cfg.get("filter")
    abstract_model_name = local_cfg.get("abstract")
    if not filter_model_name or not abstract_model_name:
        raise ValueError("config.yaml 缺少 local_model.filter 或 local_model.abstract 配置。")

    # Qwen
    qwen_cfg: Dict[str, Any] = cfg.get("qwen_api", {}) or {}
    qwen_model_name = args.qwen_model or qwen_cfg.get("model_name", "qwen-plus")
    qwen_api_key_env = args.qwen_api_key_env or qwen_cfg.get("api_key_env_var", "QWEN_API_KEY")

    _ensure_dir(args.ground_truth_dir)
    _ensure_dir(os.path.join("data", "local"))

    all_ids = _discover_ids_with_json(args.ground_truth_dir)
    if args.only_ids:
        wanted = [s.strip() for s in args.only_ids.split(",") if s.strip()]
        all_ids = [pid for pid in wanted if pid in set(all_ids)]
        if not all_ids:
            print("only_ids 过滤后没有可处理的编号。")
            return

    have_ids = _existing_jsonl_ids(args.ground_truth_dir)
    missing_ids = [pid for pid in all_ids if pid not in have_ids]

    if not missing_ids:
        print("✅ ground_truth 中已全部配对，无需补齐。")
        return

    print(f"共需补齐 {len(missing_ids)} 篇：{', '.join(missing_ids[:10])}{' ...' if len(missing_ids) > 10 else ''}")

    # 选择Qwen客户端
    if args.qwen_base_url:
        llm_client_factory = lambda: QwenDirect(api_key_env_var=qwen_api_key_env, model_name=qwen_model_name, base_url=args.qwen_base_url)
    else:
        llm_client_factory = lambda: QwenLLM(api_key_env_var=qwen_api_key_env, model_name=qwen_model_name)
    # 抑制 tokenizers 并行警告（仅进程内）
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    filled, skipped = 0, 0
    for i, paper_id in enumerate(missing_ids, 1):
        pdf_path = os.path.join(args.papers_dir, f"{paper_id}.pdf")
        if not os.path.exists(pdf_path):
            print(f"[{i}/{len(missing_ids)}] 缺失PDF，跳过：{pdf_path}")
            skipped += 1
            continue

        print(f"[{i}/{len(missing_ids)}] 处理：{os.path.basename(pdf_path)}")
        paths_out = process_one_pdf(
            pdf_path,
            output_root=os.path.join("data", "local"),
            models_root=models_root,
            filter_model_name=filter_model_name,
            abstract_model_name=abstract_model_name,
            top_n=args.topn,
            resume=args.resume,
            do_abstract=bool(args.use_abstract),
        )
        src_path = paths_out["abstract"] if (args.use_abstract and paths_out.get("abstract")) else paths_out["filtered"]

        out_jsonl_path = os.path.join(args.ground_truth_dir, f"{paper_id}.jsonl")
        # build_jsonl_from_text 内部会自行实例化 QwenLLM；这里为了复用其清洗逻辑与并发控制，直接调用
        # 注意：build_jsonl_from_text 当前不接受外部 llm 实例，但其内部使用 QwenLLM 与我们上方 client_factory 一致的配置
        created, _sk = build_jsonl_from_text(
            src_path,
            jsonl_output=out_jsonl_path,
            qwen_model_name=qwen_model_name,
            qwen_api_key_env=qwen_api_key_env,
            concurrency=max(1, int(args.concurrency)),
            resume=bool(args.resume),
            temperature=float(args.qwen_temperature),
            top_p=float(args.qwen_top_p) if args.qwen_top_p is not None else None,
            merge_output_path=None,
            granularity=args.granularity,
            min_chars=int(args.min_chars or 0),
        )
        print(f"  → 生成: {created} 行 → {out_jsonl_path}")
        filled += 1

    print(f"完成。补齐 {filled} 篇，跳过 {skipped} 篇。")


if __name__ == "__main__":
    main()


