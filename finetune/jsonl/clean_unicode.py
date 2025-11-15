#!/usr/bin/env python3
"""
清理 JSONL 文件中的 Unicode 问题字符
- 移除组合标记（combining marks）
- 规范化科学符号
- 保留正常的非ASCII字符（如°C, μ等）
"""

import json
import unicodedata
import sys
from pathlib import Path


def normalize_unicode(text: str) -> str:
    """规范化Unicode字符"""
    if not text:
        return text
    
    # 1. 移除组合标记（combining marks）
    # 使用NFD分解，然后过滤掉组合标记
    normalized = unicodedata.normalize('NFD', text)
    cleaned = ''.join(
        char for char in normalized 
        if unicodedata.category(char) != 'Mn'  # 移除组合标记
    )
    
    # 2. 规范化回NFC形式（合并字符）
    cleaned = unicodedata.normalize('NFC', cleaned)
    
    # 3. 移除控制字符（除了常见的换行、制表符）
    cleaned = ''.join(
        char for char in cleaned
        if ord(char) >= 32 or char in '\n\t\r'
    )
    
    return cleaned


def clean_jsonl_file(input_path: str, output_path: str = None) -> None:
    """清理JSONL文件"""
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_cleaned.jsonl"
    else:
        output_path = Path(output_path)
    
    print(f"读取: {input_path}")
    print(f"输出: {output_path}")
    
    cleaned_count = 0
    total_count = 0
    
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        
        for line_num, line in enumerate(fin, 1):
            total_count += 1
            try:
                # 解析JSON
                data = json.loads(line)
                
                # 清理所有字符串字段
                if isinstance(data, dict):
                    if 'messages' in data:
                        for msg in data['messages']:
                            if 'content' in msg and isinstance(msg['content'], str):
                                original = msg['content']
                                cleaned = normalize_unicode(original)
                                if original != cleaned:
                                    cleaned_count += 1
                                msg['content'] = cleaned
                
                # 写回
                fout.write(json.dumps(data, ensure_ascii=False) + '\n')
                
            except json.JSONDecodeError as e:
                print(f"⚠️  行{line_num} JSON解析错误，跳过: {e}")
                continue
            except Exception as e:
                print(f"⚠️  行{line_num} 处理错误，跳过: {e}")
                continue
    
    print(f"\n完成:")
    print(f"  总行数: {total_count}")
    print(f"  清理行数: {cleaned_count}")
    print(f"  输出文件: {output_path}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python3 clean_unicode.py <input.jsonl> [output.jsonl]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    clean_jsonl_file(input_file, output_file)


