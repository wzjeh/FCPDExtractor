#!/usr/bin/env python3
"""
全面修复所有文件中的中文
"""

import json
import os
import glob
import re

# 更全面的中文到英文映射
CHINESE_TO_ENGLISH = {
    # 反应类型
    "氢氧化": "hydrogenation",
    "直接合成": "direct synthesis", 
    "光催化降解": "photocatalytic degradation",
    "溶剂-抗溶剂沉淀": "solvent-antisolvent precipitation",
    "硝化": "nitration",
    "氢化": "hydrogenation",
    "芳香族硝化": "aromatic nitration",
    "偶联反应": "coupling reaction",
    "直接氟化": "direct fluorination",
    "酶催化合成": "enzymatic synthesis",
    
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
    "环己烯": "cyclohexene",
    "钯/氧化铝": "Pd/Al2O3",
    "超临界二氧化碳": "supercritical CO2",
    "环己烷": "cyclohexane",
    "氯苯": "chlorobenzene",
    "混合酸（硝酸/硫酸）": "mixed acid (HNO3/H2SO4)",
    "单硝基甲苯": "mononitrotoluene",
    "单硝基氯苯": "mononitrochlorobenzene",
    "硝酸-乙酸酐": "nitric acid-acetic anhydride",
    "无": "none",
    "三氟甲苯": "trifluorotoluene",
    "混合酸": "mixed acid",
    
    # 新增化学物质
    "L-谷氨酰胺": "L-glutamine",
    "乙胺": "ethylamine", 
    "谷氨酰胺酶": "glutaminase",
    "乙胺盐酸盐溶液": "ethylamine hydrochloride solution",
    "L-茶氨酸": "L-theanine",
    "2-乙基-9,10-蒽醌": "2-ethyl-9,10-anthraquinone",
    "1,2,4-三甲基苯和三辛基磷酸酯混合溶剂": "1,2,4-trimethylbenzene and trioctyl phosphate mixed solvent",
    "2-乙基蒽氢醌": "2-ethylanthrahydroquinone",
    "蒽醌": "anthraquinone",
    "硝硫混酸": "nitric-sulfuric acid mixture",
    "浓硫酸": "concentrated sulfuric acid",
    "1-硝基蒽醌": "1-nitroanthraquinone",
    "毛细管式微反应器": "capillary microreactor",
    "钯纳米催化剂": "palladium nanocatalyst",
    "乙醇-水": "ethanol-water",
    "Pd纳米颗粒": "Pd nanoparticles",
    "乙醇:水(7:3)": "ethanol:water (7:3)",
    "室温": "room temperature",
    "3-三氟甲基硝基苯": "3-trifluoromethylnitrobenzene",
    "2-乙基己醇": "2-ethylhexanol",
    "2-乙基己基硝酸酯": "2-ethylhexyl nitrate",
    "一氧化碳": "carbon monoxide",
    "氟气": "fluorine gas",
    "碳酰氟": "carbonyl fluoride",
    "硝基苯": "nitrobenzene",
    "苯胺": "aniline",
    "聚合物包埋钯": "polymer-embedded palladium",
    "过氧化氢": "hydrogen peroxide",
    "二氯甲烷": "dichloromethane",
    "铂": "platinum",
    "氮气": "nitrogen",
    
    # 反应器类型
    "椭圆挡板混合器": "elliptical baffle mixer",
    "管式反应器": "tubular reactor",
    "微通道反应器": "microchannel reactor",
    "盘管微反应器": "coil microreactor",
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
    print("全面修复所有文件中的中文...")
    
    # 获取所有reaction JSON文件
    reaction_files = glob.glob("finetune/*_reaction.json")
    
    print(f"找到 {len(reaction_files)} 个文件需要检查")
    
    success_count = 0
    for file_path in reaction_files:
        if fix_chinese_in_file(file_path):
            success_count += 1
            print(f"✓ 修复: {file_path}")
        else:
            print(f"✗ 失败: {file_path}")
    
    print(f"\n修复完成: {success_count}/{len(reaction_files)} 个文件")
    
    # 检查是否还有中文
    print("\n检查剩余中文...")
    remaining_chinese = 0
    for file_path in reaction_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        chinese_count = len(re.findall(r'[\u4e00-\u9fa5]', content))
        if chinese_count > 0:
            print(f"  {file_path}: {chinese_count} 个中文字符")
            remaining_chinese += chinese_count
    
    if remaining_chinese == 0:
        print("✓ 所有中文已清除！")
    else:
        print(f"⚠️ 还有 {remaining_chinese} 个中文字符需要手动处理")

if __name__ == "__main__":
    main()
