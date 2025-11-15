from __future__ import annotations

import os
import re
import json
import argparse
from typing import List, Dict, Any
import sys
from pathlib import Path

# 确保可以从项目根目录导入
try:
    repo_root = Path(__file__).resolve().parents[2]
    if repo_root.exists():
        sys.path.insert(0, str(repo_root))
except Exception:
    pass


def _discover_ids(ground_truth_dir: str) -> List[str]:
    ids: List[str] = []
    for fn in os.listdir(ground_truth_dir):
        if fn.endswith("_reaction_annotated.json"):
            m = re.match(r"^(\d+)_", fn)
            if m:
                ids.append(m.group(1))
    ids.sort(key=lambda x: int(x))
    return ids


def _read_candidates_from_jsonl(jsonl_path: str) -> List[str]:
    candidates: List[str] = []
    if not os.path.exists(jsonl_path):
        return candidates
    with open(jsonl_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = (line or "").strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            msgs = obj.get("messages") or []
            for m in msgs:
                if m.get("role") == "assistant":
                    content = (m.get("content") or "").strip()
                    if content:
                        candidates.append(content)
                    break
    return candidates


def _load_ground_truth_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _build_user_context(candidates: List[str]) -> str:
    # 将所有候选JSON串接，格式与 Summarized.txt 一致（每行一个JSON，行间有空行）
    # 构建类似 Summarized.txt 的格式：每个JSON一行，JSON之间有空行
    json_lines = []
    for c in candidates:
        json_lines.append(c)  # 每个JSON一行
        json_lines.append("")  # 每个JSON后加一个空行
    
    # 移除最后一个空行
    if json_lines and json_lines[-1] == "":
        json_lines.pop()
    
    combined_jsons = "\n".join(json_lines)
    
    # 使用与 summarize_document_overall 一致的提示词格式
    user_prompt = (
        "Extract the OPTIMAL condition set from the following JSON candidates. Output ONE JSON:\n"
        '{"reaction_summary":{"reaction_type":"hydrogenation","reactants":["furfural","H2","Pd/C catalyst"],'
        '"products":["furfuryl alcohol"],'
        '"conditions":[{"type":"temperature","value":"80 °C"},{"type":"residence_time","value":"5 min"},{"type":"pressure","value":"2 MPa"}],'
        '"reactor":{"type":"packed bed","inner_diameter":"5 mm"},'
        '"metrics":{"conversion":95.2,"yield":89.5,"selectivity":94.1,"unit":"%"}}}\n'
        "Choose best yield/conversion. Use null if unknown. Numbers for metrics.\n"
    )
    
    # 将候选JSON作为context，格式与 _create_prompt 一致
    return f"Context\n{combined_jsons}\n\nTask\n{user_prompt}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="将 ground_truth/{id}.jsonl 的所有候选JSON作为上下文，目标为 {id}_reaction_annotated.json，生成 {id}+json2json.jsonl（每组一条对话）。"
    )
    parser.add_argument("--ground_truth_dir", type=str, default="finetune/ground_truth")
    parser.add_argument("--source_jsonl_dir", type=str, default=None, help="候选jsonl所在目录，默认与ground_truth_dir相同")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录，默认写回 ground_truth_dir")
    parser.add_argument("--only_ids", type=str, default=None, help="仅处理这些编号（逗号分隔）")
    args = parser.parse_args()

    gt_dir = args.ground_truth_dir
    src_dir = args.source_jsonl_dir or gt_dir
    out_dir = args.output_dir or gt_dir

    _ensure_dir(out_dir)

    ids = _discover_ids(gt_dir)
    if args.only_ids:
        wanted = [s.strip() for s in args.only_ids.split(",") if s.strip()]
        ids = [pid for pid in wanted if pid in set(ids)]
        if not ids:
            print("only_ids 过滤后没有可处理的编号。")
            return

    total_written, total_skipped = 0, 0
    for i, pid in enumerate(ids, 1):
        gt_json_path = os.path.join(gt_dir, f"{pid}_reaction_annotated.json")
        src_jsonl_path = os.path.join(src_dir, f"{pid}.jsonl")
        out_path = os.path.join(out_dir, f"{pid}+json2json.jsonl")

        if not os.path.exists(src_jsonl_path):
            print(f"[{i}/{len(ids)}] 缺少候选jsonl，跳过：{src_jsonl_path}")
            total_skipped += 1
            continue
        if not os.path.exists(gt_json_path):
            print(f"[{i}/{len(ids)}] 缺少标注json，跳过：{gt_json_path}")
            total_skipped += 1
            continue

        candidates = _read_candidates_from_jsonl(src_jsonl_path)
        if not candidates:
            print(f"[{i}/{len(ids)}] 候选为空，跳过：{src_jsonl_path}")
            total_skipped += 1
            continue

        # user上下文
        user_content = _build_user_context(candidates)
        # 目标assistant内容：使用标注JSON（顶层schema）
        gt_obj = _load_ground_truth_json(gt_json_path)
        assistant_content = json.dumps(gt_obj, ensure_ascii=False, separators=(",", ":"))

        # 写入一行，system prompt 与 summarize_document_overall 一致
        messages = [
            {"role": "system", "content": "You output ONLY valid JSON. No explanations."},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
        print(f"[{i}/{len(ids)}] 生成：{out_path}（候选 {len(candidates)} 条）")
        total_written += 1

    print(f"完成。生成 {total_written} 组，跳过 {total_skipped} 组。")


if __name__ == "__main__":
    main()


