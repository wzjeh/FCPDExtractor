#!/usr/bin/env python3
"""
修复第23-50号文件中残留的中文
"""

import json
import os
import glob

# 中文到英文的映射
CHINESE_TO_ENGLISH = {
    # 反应类型
    "氢氧化": "hydrogenation",
    "直接合成": "direct synthesis", 
    "光催化降解": "photocatalytic degradation",
    "溶剂-抗溶剂沉淀": "solvent-antisolvent precipitation",
    "硝化": "nitration",
    "氢化": "hydrogenation",
    
    # 化学物质
    "氢气": "hydrogen",
    "氧气": "oxygen", 
    "水": "water",
    "甲醇": "methanol",
    "4-氯苯酚": "4-chlorophenol",
    "降解产物": "degradation products",
    "芹菜素": "apigenin",
    "二甲基亚砜": "DMSO",
    "纳米芹菜素": "nano-apigenin",
    "甲苯": "toluene",
    "硝酸": "nitric acid",
    "硫酸": "sulfuric acid",
    "硝基甲苯": "nitrotoluene",
    "浓硝酸": "concentrated nitric acid",
    "联苯": "biphenyl",
    "十四烷": "tetradecane",
    "环己基苯": "cyclohexylbenzene",
    
    # 反应器类型
    "椭圆挡板混合器": "elliptical baffle mixer",
    "管式反应器": "tubular reactor",
}

def fix_chinese_in_file(file_path):
    """修复单个文件中的中文"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 替换中文
        for chinese, english in CHINESE_TO_ENGLISH.items():
            content = content.replace(f'"{chinese}"', f'"{english}"')
        
        # 写回文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return True
    except Exception as e:
        print(f"修复 {file_path} 失败: {e}")
        return False

def main():
    print("修复第23-50号文件中的中文...")
    
    # 获取第23-50号的reaction JSON文件
    reaction_files = []
    for i in range(23, 51):
        file_path = f"finetune/{i}_reaction.json"
        if os.path.exists(file_path):
            reaction_files.append(file_path)
    
    print(f"找到 {len(reaction_files)} 个文件需要检查")
    
    success_count = 0
    for file_path in reaction_files:
        if fix_chinese_in_file(file_path):
            success_count += 1
            print(f"✓ 修复: {file_path}")
        else:
            print(f"✗ 失败: {file_path}")
    
    print(f"\n修复完成: {success_count}/{len(reaction_files)} 个文件")

if __name__ == "__main__":
    main()
