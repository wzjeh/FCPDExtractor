from __future__ import annotations

import os
import re
import argparse
from typing import List, Dict, Any, Tuple

# 复用现有逻辑，避免重复实现
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
)


def _discover_annotated_ids(ground_truth_dir: str) -> List[str]:
    annotated_ids: List[str] = []
    if not os.path.isdir(ground_truth_dir):
        return annotated_ids
    for fn in os.listdir(ground_truth_dir):
        # 形如 "114_reaction_annotated.json"
        m = re.match(r"^(\d+)_", fn)
        if m and fn.endswith("_reaction_annotated.json"):
            annotated_ids.append(m.group(1))
    annotated_ids.sort(key=lambda x: int(x))
    return annotated_ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="针对 ground_truth 中已标注的论文编号，逐篇从 finetune/papers/{id}.pdf 提取并各自生成 {id}.jsonl"
    )
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--ground_truth_dir", type=str, default="finetune/ground_truth")
    parser.add_argument("--papers_dir", type=str, default="finetune/papers")
    parser.add_argument("--jsonl_output_dir", type=str, default="finetune/jsonl")
    parser.add_argument("--topn", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="跳过前N篇后再开始处理（结合 --limit 可实现分批）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="仅处理前N篇（按编号排序或 only_ids 的顺序）",
    )
    parser.add_argument(
        "--only_ids",
        type=str,
        default=None,
        help="仅处理这些编号（逗号分隔，如: 114,1,82）。若不提供则处理 ground_truth 里全部编号。",
    )
    parser.add_argument(
        "--use_abstract",
        action="store_true",
        help="是否在本地筛选后先做段落摘要，再进行Qwen提取",
    )
    parser.add_argument(
        "--write_merged",
        action="store_true",
        help="为每篇论文在 data/local/<paper>/ 下写出 Merged_Overall.json（整篇合并结果）",
    )
    parser.add_argument(
        "--granularity",
        type=str,
        default="sentence",
        choices=["sentence", "paragraph"],
        help="Qwen 提取粒度：句子或段落",
    )
    parser.add_argument(
        "--min_chars", type=int, default=0, help="丢弃短于该字符数的上下文"
    )
    parser.add_argument(
        "--qwen_model",
        type=str,
        default=None,
        help="覆盖 config.yaml 中的 qwen_api.model_name",
    )
    parser.add_argument(
        "--qwen_api_key_env",
        type=str,
        default=None,
        help="覆盖 config.yaml 中的 qwen_api.api_key_env_var",
    )
    parser.add_argument(
        "--qwen_temperature",
        type=float,
        default=0.0,
        help="Qwen 采样温度",
    )
    parser.add_argument(
        "--qwen_top_p",
        type=float,
        default=0.6,
        help="Qwen top_p",
    )
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
    qwen_api_key_env = args.qwen_api_key_env or qwen_cfg.get(
        "api_key_env_var", "QWEN_API_KEY"
    )

    annotated_ids = _discover_annotated_ids(args.ground_truth_dir)
    if not annotated_ids:
        print("未在 ground_truth 目录发现 *_reaction_annotated.json。")
        return

    # 如果指定了 only_ids，则只取交集顺序化处理
    if args.only_ids:
        wanted = [s.strip() for s in args.only_ids.split(",") if s.strip()]
        wanted_set = set(wanted)
        gt_set = set(annotated_ids)
        selected = [pid for pid in wanted if pid in gt_set]
        missing_from_gt = [pid for pid in wanted if pid not in gt_set]
        if missing_from_gt:
            print(f"警告：以下编号未在 ground_truth 中找到标注文件，将被忽略：{', '.join(missing_from_gt)}")
        if not selected:
            print("only_ids 过滤后没有可处理的编号。")
            return
        annotated_ids = selected

    # 先偏移，再截断
    if args.offset:
        annotated_ids = annotated_ids[int(args.offset) :]
    # 若给定 limit，则只取前N（偏移后）
    if args.limit is not None:
        annotated_ids = annotated_ids[: max(0, int(args.limit))]

    _ensure_dir(args.jsonl_output_dir)
    output_root = os.path.join("data", "local")
    _ensure_dir(output_root)

    # 抑制 tokenizers 并行警告（仅进程内）
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    total_created, total_skipped, total_processed, total_missing = 0, 0, 0, 0
    print(
        f"共发现 {len(annotated_ids)} 篇标注文件。TopN={args.topn} 并发={args.concurrency} resume={args.resume}"
    )
    for idx, paper_id in enumerate(annotated_ids, 1):
        pdf_path = os.path.join(args.papers_dir, f"{paper_id}.pdf")
        if not os.path.exists(pdf_path):
            print(f"[{idx}/{len(annotated_ids)}] 缺失PDF: {pdf_path}")
            total_missing += 1
            continue

        print(f"[{idx}/{len(annotated_ids)}] 处理: {os.path.basename(pdf_path)}")
        paths_out = process_one_pdf(
            pdf_path,
            output_root=output_root,
            models_root=models_root,
            filter_model_name=filter_model_name,
            abstract_model_name=abstract_model_name,
            top_n=args.topn,
            resume=args.resume,
            do_abstract=bool(args.use_abstract),
        )

        src_path = (
            paths_out["abstract"]
            if (args.use_abstract and paths_out.get("abstract"))
            else paths_out["filtered"]
        )
        out_jsonl_path = os.path.join(args.jsonl_output_dir, f"{paper_id}.jsonl")

        merge_path = None
        if args.write_merged:
            merge_path = os.path.join(os.path.dirname(src_path), "Merged_Overall.json")

        created, skipped = build_jsonl_from_text(
            src_path,
            jsonl_output=out_jsonl_path,
            qwen_model_name=qwen_model_name,
            qwen_api_key_env=qwen_api_key_env,
            concurrency=max(1, int(args.concurrency)),
            resume=bool(args.resume),
            temperature=float(args.qwen_temperature),
            top_p=float(args.qwen_top_p) if args.qwen_top_p is not None else None,
            merge_output_path=merge_path,
            granularity=args.granularity,
            min_chars=int(args.min_chars or 0),
        )
        print(f"  → 生成: {created} 行，跳过: {skipped} 行 → {out_jsonl_path}")
        total_created += created
        total_skipped += skipped
        total_processed += 1

    print(
        f"完成。已处理 {total_processed} 篇，缺失 {total_missing} 篇。合计新增 {total_created} 行，跳过 {total_skipped} 行。"
    )


if __name__ == "__main__":
    main()


