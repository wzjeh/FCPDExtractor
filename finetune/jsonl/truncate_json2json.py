#!/usr/bin/env python3
"""
截断 json2json.jsonl 中过长的 user content，确保不超过 token 限制
"""
import json
import argparse
import os


def truncate_user_content(user_content: str, max_chars: int = 8000) -> str:
    """
    截断 user content，保留 Context 部分的前 N 个 JSON 候选
    max_chars: 最大字符数（约 2000 tokens，按 1 token ≈ 4 字符）
    """
    if len(user_content) <= max_chars:
        return user_content
    
    # 分割 Context 和 Task 部分
    parts = user_content.split("\n\nTask\n", 1)
    if len(parts) != 2:
        return user_content[:max_chars]  # 如果格式不对，直接截断
    
    context_part, task_part = parts
    
    # 提取 Context 行后的内容（JSON 列表）
    context_lines = context_part.split("\n", 1)
    if len(context_lines) < 2:
        return user_content[:max_chars]
    
    header = context_lines[0] + "\n"  # "Context" 行
    json_candidates = context_lines[1]
    
    # 按空行分割 JSON（每个 JSON 后有一个空行）
    json_blocks = json_candidates.split("\n\n")
    json_blocks = [j for j in json_blocks if j.strip()]  # 移除空块
    
    # 逐步添加 JSON，直到接近限制
    truncated_jsons = []
    current_length = len(header) + len("\n\nTask\n") + len(task_part)
    
    for json_block in json_blocks:
        # 加上这个 JSON 和一个空行后的长度
        block_with_newline = json_block + "\n\n"
        if current_length + len(block_with_newline) > max_chars:
            break
        truncated_jsons.append(json_block)
        current_length += len(block_with_newline)
    
    if not truncated_jsons:
        # 如果第一个 JSON 就超了，至少保留第一个（截断）
        if json_blocks:
            first_json = json_blocks[0]
            available = max_chars - current_length
            if available > 100:  # 至少保留 100 字符
                truncated_jsons.append(first_json[:available])
    
    # 重新组合
    truncated_context = header + "\n\n".join(truncated_jsons) + "\n\nTask\n" + task_part
    return truncated_context


def main():
    parser = argparse.ArgumentParser(description="截断 json2json.jsonl 中过长的 user content")
    parser.add_argument('--input', type=str, default='finetune/jsonl/json2json.jsonl',
                        help='输入文件路径')
    parser.add_argument('--output', type=str, default=None,
                        help='输出文件路径（默认覆盖输入文件）')
    parser.add_argument('--max_chars', type=int, default=8000,
                        help='最大字符数（默认 8000，约 2000 tokens）')
    args = parser.parse_args()
    
    output_path = args.output or args.input
    backup_path = args.input + '.backup'
    
    # 备份原文件
    import shutil
    shutil.copy2(args.input, backup_path)
    print(f"已备份原文件到: {backup_path}")
    
    total_lines = 0
    truncated_count = 0
    max_before = 0
    max_after = 0
    
    with open(args.input, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            total_lines += 1
            obj = json.loads(line)
            user_content = obj['messages'][1]['content']
            original_len = len(user_content)
            max_before = max(max_before, original_len)
            
            if original_len > args.max_chars:
                truncated_content = truncate_user_content(user_content, args.max_chars)
                obj['messages'][1]['content'] = truncated_content
                truncated_count += 1
                new_len = len(truncated_content)
                max_after = max(max_after, new_len)
                print(f"  样本 #{total_lines}: {original_len} → {new_len} 字符 ({new_len//4} tokens)")
            else:
                max_after = max(max_after, original_len)
            
            fout.write(json.dumps(obj, ensure_ascii=False) + '\n')
    
    print(f"\n完成！")
    print(f"总样本数: {total_lines}")
    print(f"截断的样本数: {truncated_count}")
    print(f"截断前最大长度: {max_before} 字符 ({max_before//4} tokens)")
    print(f"截断后最大长度: {max_after} 字符 ({max_after//4} tokens)")
    print(f"输出文件: {output_path}")


if __name__ == '__main__':
    main()













