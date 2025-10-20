# 📝 Notebook步骤5更新代码（双模式）
# 在线LLM可选：标准5步 或 快速直达
# 本地LLM：强制使用标准5步

print("🚀 步骤 5/5: 抽象和总结...")
print("=" * 50)

# 🎛️ 在线LLM快速模式开关（本地LLM强制为False）
USE_FAST_MODE = True if ENGINE in ['qwen', 'gemini'] else False

abstract_files = []
summarized_files = []
overall_files = []
impact_files = []

for i, filter_file in enumerate(filter_files, 1):
    print(f"\n📄 处理文件 {i}/{len(filter_files)}: {os.path.basename(filter_file)}")

    # 🚀 在线LLM快速模式：直接从原始txt提取Overall+Impact
    if USE_FAST_MODE and ENGINE in ['qwen', 'gemini']:
        print(f"  ⚡ {ENGINE.upper()}快速模式：跳过步骤3-5，直接提取（2次API调用）")
        # 使用步骤1的原始txt（完整文本）
        original_txt = output_files[i-1]
        res_fast = engine.process_text_file_comprehensive(original_txt, mode='fast')
        
        overall_path = res_fast.get('summarized_overall')
        impact_path = res_fast.get('impact_analysis')
        
        if overall_path:
            overall_files.append(overall_path)
            print(f"  🧩 整篇汇总 → {overall_path}")
        if impact_path:
            impact_files.append(impact_path)
            print(f"  📊 影响因素 → {impact_path}")
        
        # 归档
        try:
            base = os.path.splitext(os.path.basename(filter_file))[0].replace('Embedding_','').replace('_Filtered','')
            dest_dir = os.path.join(main_output_dir, '..', ENGINE_NAME, base)
            os.makedirs(dest_dir, exist_ok=True)
            for p in [overall_path, impact_path]:
                if p and os.path.exists(p):
                    shutil.copy2(p, os.path.join(dest_dir, os.path.basename(p)))
        except Exception:
            pass
    
    else:
        # 标准5步流程（本地LLM 或 在线LLM标准模式）
        # 抽象
        res_abs = engine.process_text_file_comprehensive(filter_file, mode='abstract')
        abstract_file_path = list(res_abs.values())[0] if res_abs else ''
        abstract_files.append(abstract_file_path)
        print(f"  ✅ 抽象完成 → {abstract_file_path}")

        # 总结（优先使用抽象后的文本）
        input_for_sum = abstract_file_path if os.path.exists(abstract_file_path) else filter_file
        res_sum = engine.process_text_file_comprehensive(input_for_sum, mode='summarize')
        summarized_file_path = list(res_sum.values())[0] if res_sum else ''
        summarized_files.append(summarized_file_path)
        print(f"  ✅ 逐段总结 → {summarized_file_path}")

        # Overall & Impact
        overall_path = res_sum.get('summarized_overall') if isinstance(res_sum, dict) else None
        impact_path = res_sum.get('impact_analysis') if isinstance(res_sum, dict) else None
        if overall_path:
            overall_files.append(overall_path)
            print(f"  🧩 整篇汇总 → {overall_path}")
        if impact_path:
            impact_files.append(impact_path)
            print(f"  📊 影响因素 → {impact_path}")

        # 归档
        try:
            base = os.path.splitext(os.path.basename(filter_file))[0].replace('Embedding_','').replace('_Filtered','')
            dest_dir = os.path.join(main_output_dir, '..', ENGINE_NAME, base)
            os.makedirs(dest_dir, exist_ok=True)
            for p in [abstract_file_path, summarized_file_path, overall_path, impact_path]:
                if p and os.path.exists(p):
                    shutil.copy2(p, os.path.join(dest_dir, os.path.basename(p)))
        except Exception:
            pass

print(f"\n🎉 抽象和总结完成！")
print(f"📊 最终产物汇总:")
if USE_FAST_MODE:
    print(f"  ⚡ 快速模式（仅Overall+Impact）")
    print(f"  - 整篇汇总(Overall): {len(overall_files)} 个")
    print(f"  - 影响因素(Impact): {len(impact_files)} 个")
else:
    print(f"  📚 标准模式（完整5步）")
    print(f"  - 抽象文件: {len(abstract_files)} 个")
    print(f"  - 逐段总结: {len(summarized_files)} 个")
    print(f"  - 整篇汇总(Overall): {len(overall_files)} 个")
    print(f"  - 影响因素(Impact): {len(impact_files)} 个")

