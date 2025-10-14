import pandas as pd
import os
import sys
sys.path.append('Text Parser')
sys.path.append('Text Extraction')
from PDF_Unified_Processor import save_contents_to_specific_folders
from TXT_Processing import process_text_file_for_processing
from Embedding_and_Similarity import process_text_file_for_embedding
from Unified_Text_Processor import (
    process_text_file_for_filter, process_text_file_for_abstract, process_text_file_for_summerized,
    process_text_file_for_filter, process_text_file_for_abstract, 
    process_text_file_for_summerized
)

print("🚀 启动 OSSExtractor 表面合成参数提取工具")
print("=" * 60)

pdf_files = [
"/Users/zhaowenyuan/Projects/FCPDExtractor/Data/papers/101021acsoprd7b00291.pdf"
]
base_output_dir = '/Users/zhaowenyuan/Projects/FCPDExtractor/Data'  

print("📄 步骤 1/5: PDF转文本处理...")
output_files = save_contents_to_specific_folders(pdf_files, base_output_dir)
print("✅ PDF转文本完成")

print("\n📝 步骤 2/5: 文本预处理...")
processed_files = []
total_filtered_count = 0
for file_path in output_files:
    print(f"处理文件: {os.path.basename(file_path)}")
    processed_file_path, filtered_count = process_text_file_for_processing(file_path)
    processed_files.append(processed_file_path)
    total_filtered_count += filtered_count
print(f"✅ 文本预处理完成，过滤了 {total_filtered_count} 个段落")

print("\n🔍 步骤 3/5: 嵌入和相似度计算...")
embedding_files = []
for file_path in processed_files:
    print(f"处理文件: {os.path.basename(file_path)}")
    embedding_file_path = process_text_file_for_embedding(file_path)
    embedding_files.append(embedding_file_path)
print("✅ 嵌入和相似度计算完成")

print("\n🤖 步骤 4/5: LLM内容过滤...")
print("💡 使用nous-hermes-llama2-13b模型进行智能过滤")
filter_files = []
for file_path in embedding_files:
    print(f"处理文件: {os.path.basename(file_path)}")
    filter_file_path = process_text_file_for_filter(file_path)
    filter_files.append(filter_file_path)
print("✅ LLM内容过滤完成")

print("\n📊 步骤 5/5: 抽象和总结...")
print("💡 使用nous-hermes-llama2-13b模型进行抽象和总结")
abstract_files = []
summarized_files = []
for file_path in filter_files:
    print(f"处理文件: {os.path.basename(file_path)}")
    abstract_file_path = process_text_file_for_abstract(file_path)
    summerized_file_path = process_text_file_for_summerized(file_path)
    abstract_files.append(abstract_file_path)
    summarized_files.append(summerized_file_path)
print("✅ 抽象和总结完成")

print("\n🎉 所有处理步骤完成！")
print("=" * 60)

# 显示最终结果
print("\n📊 处理结果总结:")
print("=" * 30)
print(f"📁 原始文本文件: {len(output_files)} 个")
print(f"📁 预处理文件: {len(processed_files)} 个")
print(f"📁 嵌入文件: {len(embedding_files)} 个")
print(f"📁 过滤文件: {len(filter_files)} 个")
print(f"📁 抽象文件: {len(abstract_files)} 个")
print(f"📁 总结文件: {len(summarized_files)} 个")

print(f"\n🎯 最终输出文件:")
for i, file in enumerate(summarized_files, 1):
    if os.path.exists(file):
        with open(file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        print(f"  {i}. {os.path.basename(file)} ({len(lines)} 行)")
    else:
        print(f"  {i}. {os.path.basename(file)} (文件不存在)")

print(f"\n✅ 处理完成！共处理了 {len(pdf_files)} 个PDF文件")



