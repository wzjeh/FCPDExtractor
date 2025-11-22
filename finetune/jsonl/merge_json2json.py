#!/usr/bin/env python3
"""
合并所有 {id}+json2json.jsonl 文件为一个 json2json.jsonl
"""
import os
import glob
import argparse
from pathlib import Path


def merge_json2json_files(source_dir: str, output_path: str) -> None:
    """合并所有 *+json2json.jsonl 文件"""
    pattern = os.path.join(source_dir, "*+json2json.jsonl")
    files = sorted(glob.glob(pattern))
    
    if not files:
        print(f"未找到 *+json2json.jsonl 文件（在 {source_dir}）")
        return
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    total_lines = 0
    with open(output_path, 'w', encoding='utf-8') as fout:
        for file_path in files:
            file_name = os.path.basename(file_path)
            count = 0
            with open(file_path, 'r', encoding='utf-8') as fin:
                for line in fin:
                    line = line.strip()
                    if line:  # 跳过空行
                        fout.write(line + '\n')
                        count += 1
                        total_lines += 1
            print(f"  合并: {file_name} ({count} 行)")
    
    print(f"\n完成！合并了 {len(files)} 个文件，共 {total_lines} 行 → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="合并所有 {id}+json2json.jsonl 文件")
    parser.add_argument('--source_dir', type=str, default='finetune/ground_truth',
                        help='源目录（包含 *+json2json.jsonl 文件）')
    parser.add_argument('--output', type=str, default='finetune/jsonl/json2json.jsonl',
                        help='输出文件路径')
    args = parser.parse_args()
    
    merge_json2json_files(args.source_dir, args.output)


if __name__ == '__main__':
    main()












