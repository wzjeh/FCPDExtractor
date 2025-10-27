#!/usr/bin/env python3
"""
微调数据集生成脚本 - 处理第81-118号PDF文件
使用 qwen-plus-2025-07-28 模型

功能：
使用 qwen-plus-2025-07-28 模型直接生成：
- reaction_summary：反应参数摘要（JSON格式）
- impact_analysis：操作参数对性能的影响关系

输出：
每个 PDF 对应两个文件，保存到 finetune/ 文件夹：
- {number}_reaction.json：reaction_summary
- {number}_impact.txt：impact_analysis（纯文本）

前提条件：
papers 文件夹中的 PDF 应已按数字命名（81.pdf, 82.pdf, ..., 118.pdf）
"""

import os
import json
import glob
from pathlib import Path

from core.text_utils import extract_text_from_pdf
from core.models.qwen_llm import QwenLLM


def create_reaction_summary_prompt(pdf_text: str) -> str:
    """
    构建用于生成 reaction_summary 的 prompt
    """
    # 限制文本长度，避免超出模型上下文限制
    max_chars = 20000
    if len(pdf_text) > max_chars:
        pdf_text = pdf_text[:max_chars] + "\n\n[文本已截断...]"
    
    prompt = f"""你是一位化学工程专家，专门从事流动化学和过程开发研究。

请仔细阅读以下科学文献全文，提取并总结其中最重要的流动化学反应信息。

要求：
1. 提取文献中描述的**最优实验条件**（产率/转化率最高的条件）
2. 如果文献中有多个反应实验，选择最重要或最优的一个
3. 对于未明确提及的字段，使用 null
4. 数值型字段（如转化率、产率、选择性）使用数字，不要用字符串
5. 输出**纯JSON格式**，不要包含任何解释、注释或markdown标记

输出JSON格式：
{{
  "reaction_summary": {{
    "reaction_type": "反应类型（如氢化、氧化、偶联等）",
    "reactants": [
      {{"name": "反应物名称", "role": "reactant"}},
      {{"name": "催化剂名称", "role": "catalyst"}},
      {{"name": "溶剂名称", "role": "solvent"}}
    ],
    "products": [
      {{"name": "产物名称", "yield_optimal": 95.0, "unit": "%"}}
    ],
    "conditions": [
      {{"type": "temperature", "value": "80 °C"}},
      {{"type": "residence_time", "value": "5 min"}},
      {{"type": "flow_rate_total", "value": "0.5 mL/min"}},
      {{"type": "pressure", "value": "2 MPa"}}
    ],
    "reactor": {{
      "type": "反应器类型（如packed bed, coil, tubular等）",
      "inner_diameter": "内径（如5 mm）"
    }},
    "metrics": {{
      "conversion": 95.0,
      "yield": 89.5,
      "selectivity": 94.0,
      "unit": "%"
    }}
  }}
}}

文献全文：
{pdf_text}

请输出JSON："""
    
    return prompt


def create_impact_analysis_prompt(pdf_text: str) -> str:
    """
    构建用于生成影响因素分析的 prompt
    """
    # 限制文本长度
    max_chars = 20000
    if len(pdf_text) > max_chars:
        pdf_text = pdf_text[:max_chars] + "\n\n[文本已截断...]"
    
    prompt = f"""你是一位化学工程专家，专门分析流动化学过程中的因果关系。

请从以下文献中提取**操作参数对反应性能的影响关系**。

示例：
文献段落："停留时间是一个重要参数。更长的停留时间会导致TFMB的转化率更高。产物选择性几乎保持不变。"

输出格式：
residence_time | conversion of TFMB | increase
residence_time | product selectivity | unchanged

要求：
1. 每行格式：因素 | 性能指标 | 影响方向
2. 影响方向只能是：increase（增加）、decrease（降低）、unchanged（不变）
3. 只提取文献中**明确描述的因果关系**
4. 关注的因素包括：temperature, pressure, flow_rate, residence_time, catalyst_loading, reactant_concentration 等
5. 关注的性能指标包括：conversion, yield, selectivity, purity 等
6. 不要输出表头，直接输出数据行
7. 每行一个因果关系

文献全文：
{pdf_text}

请输出影响因素分析（每行一个关系，格式：因素 | 性能指标 | 影响方向）："""
    
    return prompt


def parse_impact_response(response: str) -> list:
    """
    解析影响因素分析的LLM响应
    
    Args:
        response: LLM原始响应
    
    Returns:
        影响因素列表，每个元素包含 factor, metric, direction
    """
    items = []
    
    for line in response.splitlines():
        line = line.strip()
        if not line or '|' not in line:
            continue
        
        # 跳过表头行
        low = line.lower()
        if 'factor' in low and 'metric' in low:
            continue
        
        parts = [p.strip() for p in line.split('|')]
        if len(parts) < 2:
            continue
        
        # 跳过空行
        if not parts[0] or not parts[1] or parts[0] == '-':
            continue
        
        # 兼容2列和3列格式
        if len(parts) == 2:
            # 两列格式：Factor | Metric（从Metric中提取Direction）
            factor = parts[0]
            metric_with_dir = parts[1].lower()
            
            # 尝试从metric字段提取方向
            if 'increase' in metric_with_dir or 'higher' in metric_with_dir:
                direction = 'increase'
                metric = parts[1].split()[0]
            elif 'decrease' in metric_with_dir or 'lower' in metric_with_dir:
                direction = 'decrease'
                metric = parts[1].split()[0]
            elif 'unchange' in metric_with_dir:
                direction = 'unchanged'
                metric = parts[1].split()[0]
            else:
                direction = None
                metric = parts[1]
        else:
            # 三列格式：Factor | Metric | Direction
            factor = parts[0]
            metric = parts[1]
            direction = parts[2].lower()
            
            # 标准化方向
            if 'increase' in direction or 'higher' in direction or 'improve' in direction or 'enhance' in direction:
                direction = 'increase'
            elif 'decrease' in direction or 'lower' in direction or 'reduce' in direction or 'inhibit' in direction:
                direction = 'decrease'
            elif 'unchange' in direction or 'unchanged' in direction or 'no effect' in direction:
                direction = 'unchanged'
            else:
                direction = None
        
        items.append({
            'factor': factor,
            'metric': metric,
            'direction': direction
        })
    
    return items


def generate_impact_analysis(pdf_text: str, llm: QwenLLM) -> list:
    """
    从PDF文本生成影响因素分析
    
    Args:
        pdf_text: PDF提取的文本
        llm: QwenLLM实例
    
    Returns:
        影响因素列表
    """
    print("  [3/3] 生成影响因素分析...")
    
    prompt = create_impact_analysis_prompt(pdf_text)
    
    try:
        response = llm.generate(prompt, max_tokens=800, temp=0.2)
        print(f"  ✓ Impact生成完成，响应长度: {len(response)} 字符")
        
        # 解析响应
        items = parse_impact_response(response)
        print(f"  ✓ 解析出 {len(items)} 个影响因素")
        
        return items
        
    except Exception as e:
        print(f"  ✗ Impact生成失败: {e}")
        return []


def generate_complete_dataset(pdf_path: str, llm: QwenLLM) -> dict:
    """
    从PDF生成完整的数据集（包括reaction_summary和impact_analysis）
    
    Args:
        pdf_path: PDF文件路径
        llm: QwenLLM实例
    
    Returns:
        包含reaction_summary和impact_analysis的字典
    """
    print(f"\n处理: {os.path.basename(pdf_path)}")
    
    # 步骤1: 提取PDF文本
    print("  [1/3] 提取PDF文本...")
    try:
        pdf_text = extract_text_from_pdf(pdf_path)
        print(f"  ✓ 提取完成，文本长度: {len(pdf_text)} 字符")
    except Exception as e:
        print(f"  ✗ PDF提取失败: {e}")
        return None
    
    # 步骤2: 生成reaction_summary
    print("  [2/3] 生成reaction_summary...")
    prompt = create_reaction_summary_prompt(pdf_text)
    
    reaction_summary = None
    try:
        # 使用较高的max_tokens以获取完整的JSON
        response = llm.generate(prompt, max_tokens=1500, temp=0.1)
        print(f"  ✓ 生成完成，响应长度: {len(response)} 字符")
        
        # 清理响应：去除可能的markdown标记
        cleaned_response = response.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.startswith("```"):
            cleaned_response = cleaned_response[3:]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]
        cleaned_response = cleaned_response.strip()
        
        # 提取JSON部分
        start_idx = cleaned_response.find('{')
        end_idx = cleaned_response.rfind('}')
        if start_idx != -1 and end_idx != -1:
            json_str = cleaned_response[start_idx:end_idx+1]
            reaction_summary = json.loads(json_str)
            print("  ✓ Reaction summary解析成功")
        else:
            print("  ✗ 无法找到有效的JSON结构")
            
    except json.JSONDecodeError as e:
        print(f"  ✗ JSON解析失败: {e}")
        print(f"  原始响应: {response[:500]}")
    except Exception as e:
        print(f"  ✗ 生成失败: {e}")
    
    # 步骤3: 生成impact_analysis
    impact_analysis = generate_impact_analysis(pdf_text, llm)
    
    # 组合结果
    if reaction_summary is None and not impact_analysis:
        return None
    
    result = {}
    if reaction_summary:
        result.update(reaction_summary)
    
    if impact_analysis:
        result['impact_analysis'] = impact_analysis
    
    return result


def save_separate_files(base_name: str, output_dir: str, reaction_summary: dict, impact_analysis: list) -> bool:
    """
    分别保存 reaction_summary 和 impact_analysis 到独立文件
    
    Args:
        base_name: 基础文件名（例如 "81", "82"）
        output_dir: 输出目录
        reaction_summary: reaction_summary 字典
        impact_analysis: impact_analysis 列表
    
    Returns:
        是否成功
    """
    success = True
    
    # 保存 reaction_summary 为 JSON
    if reaction_summary:
        reaction_file = os.path.join(output_dir, f"{base_name}_reaction.json")
        try:
            with open(reaction_file, 'w', encoding='utf-8') as f:
                json.dump(reaction_summary, f, ensure_ascii=False, indent=2)
            print(f"  ✓ Reaction 已保存: {reaction_file}")
        except Exception as e:
            print(f"  ✗ Reaction 保存失败: {e}")
            success = False
    
    # 保存 impact_analysis 为纯文本
    if impact_analysis:
        impact_file = os.path.join(output_dir, f"{base_name}_impact.txt")
        try:
            with open(impact_file, 'w', encoding='utf-8') as f:
                f.write("# Influence Factor Analysis\n\n")
                f.write("| Factor | Metric | Direction |\n")
                f.write("|--------|--------|-----------|\n")
                for item in impact_analysis:
                    factor = item.get('factor', '-')
                    metric = item.get('metric', '-')
                    direction = item.get('direction', '-') or '-'
                    f.write(f"| {factor} | {metric} | {direction} |\n")
            print(f"  ✓ Impact 已保存: {impact_file}")
        except Exception as e:
            print(f"  ✗ Impact 保存失败: {e}")
            success = False
    
    return success


def main():
    """
    主函数：
    批量处理 papers 文件夹中的第81-118号 PDF
    为每个 PDF 生成两个独立文件：
    - {number}_reaction.json：reaction_summary
    - {number}_impact.txt：impact_analysis
    
    前提：papers 文件夹中的 PDF 已按数字命名（81.pdf, 82.pdf, ..., 118.pdf）
    """
    print("=" * 60)
    print("微调数据集生成工具 - 处理第81-118号PDF")
    print("使用模型: qwen-plus-2025-07-28")
    print("=" * 60)
    
    # 步骤1: 获取第81-118号PDF文件
    papers_dir = "data/papers"
    pdf_files = []
    
    for i in range(81, 119):  # 81到118
        pdf_path = os.path.join(papers_dir, f"{i}.pdf")
        if os.path.exists(pdf_path):
            pdf_files.append(pdf_path)
    
    if not pdf_files:
        print(f"\n未在 {papers_dir} 中找到第81-118号 PDF 文件")
        return
    
    # 按数字排序
    pdf_files = sorted(pdf_files, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    print(f"\n找到 {len(pdf_files)} 个 PDF 文件 (第81-118号)")
    
    # 步骤2: 初始化qwen-plus-2025-07-28模型
    print("\n初始化 qwen-plus-2025-07-28 模型...")
    try:
        llm = QwenLLM(api_key_env_var="QWEN_API_KEY", model_name="qwen-plus-2025-07-28")
        print("✓ 模型初始化成功")
    except Exception as e:
        print(f"✗ 模型初始化失败: {e}")
        print("请确保已设置 QWEN_API_KEY 环境变量")
        return
    
    # 创建输出目录
    output_dir = "finetune"
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录: {output_dir}/")
    
    # 步骤3: 批量处理
    print("\n" + "=" * 60)
    print("开始生成数据集...")
    print("=" * 60)
    
    success_count = 0
    fail_count = 0
    
    for pdf_path in pdf_files:
        pdf_name = os.path.basename(pdf_path)
        base_name = os.path.splitext(pdf_name)[0]
        
        # 检查文件是否已存在
        reaction_file = os.path.join(output_dir, f"{base_name}_reaction.json")
        impact_file = os.path.join(output_dir, f"{base_name}_impact.txt")
        
        if os.path.exists(reaction_file) and os.path.exists(impact_file):
            print(f"\n跳过 {pdf_name} (已存在)")
            continue
        
        # 生成完整数据集（reaction_summary + impact_analysis）
        result = generate_complete_dataset(pdf_path, llm)
        
        if result:
            # 分别保存两个文件
            reaction_summary = result.get('reaction_summary')
            impact_analysis = result.get('impact_analysis', [])
            
            if save_separate_files(base_name, output_dir, reaction_summary, impact_analysis):
                success_count += 1
            else:
                fail_count += 1
        else:
            fail_count += 1
    
    # 总结
    print("\n" + "=" * 60)
    print(f"处理完成！")
    print(f"成功: {success_count} 个 PDF")
    print(f"失败: {fail_count} 个 PDF")
    print(f"输出目录: {output_dir}/")
    print(f"  - 每个 PDF 对应 2 个文件：")
    print(f"    * {{number}}_reaction.json")
    print(f"    * {{number}}_impact.txt")
    print("=" * 60)


if __name__ == "__main__":
    main()



