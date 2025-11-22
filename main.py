import argparse
import glob
import os
import sys
import yaml
import shutil
from typing import List

from core.text_utils import extract_text_from_pdf, write_text
from core.embedding import run_embedding_selection
from core.models.gemini_llm import GeminiLLM
from core.models.qwen_llm import QwenLLM
from core.processor import UnifiedTextProcessor
from core.local_pipeline import LocalPipeline
from evaluation.metrics import calculate_metrics


def iter_inputs(input_dir: str, limit: int | None) -> List[str]:
    files = sorted(glob.glob(os.path.join(input_dir, '*.*')))
    files = [f for f in files if f.lower().endswith(('.txt', '.pdf'))]
    return files[:limit] if limit else files


def load_config(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def ensure_txt(fp: str, output_dir: str) -> str:
    if fp.lower().endswith('.txt'):
        return fp
    base = os.path.splitext(os.path.basename(fp))[0]
    out_dir = os.path.join(output_dir, base)
    os.makedirs(out_dir, exist_ok=True)
    out_txt = os.path.join(out_dir, f"{base}.txt")
    text = extract_text_from_pdf(fp)
    write_text(text, out_txt)
    return out_txt


def main() -> None:
    parser = argparse.ArgumentParser(description="FCPDExtractor - New Architecture CLI")
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--input_dir', type=str, help='输入目录（.txt 或 .pdf）')
    parser.add_argument('--output_dir', type=str, help='输出目录')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--engine', type=str, choices=['qwen','gemini','local'])
    parser.add_argument('--mode', type=str, default='comprehensive', choices=['filter','abstract','summarize','comprehensive','evaluate'])
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg.get('paths', {})
    input_dir = args.input_dir or paths.get('papers_dir', 'data/papers')
    output_dir = args.output_dir or paths.get('output_dir', 'data')
    os.makedirs(output_dir, exist_ok=True)

    engine_choice = args.engine or cfg.get('engine', 'qwen')
    if engine_choice == 'local':
        local_cfg = cfg.get('local_model', {})
        engine = LocalPipeline(
            model_name=None,  # 使用分阶段配置
            model_path=local_cfg.get('path', 'models/'),
            filter_model=local_cfg.get('filter'),
            abstract_model=local_cfg.get('abstract'),
            summarize_model=local_cfg.get('summarize'),
            overall_model=local_cfg.get('overall'),
            impact_model=local_cfg.get('impact'),
            finetuned_trigger_name=local_cfg.get('finetuned_trigger_name', 'My_Finetuned_Model'),
        )
    elif engine_choice == 'qwen':
        qcfg = cfg.get('qwen_api', {})
        llm = QwenLLM(api_key_env_var=qcfg.get('api_key_env_var', 'QWEN_API_KEY'), 
                      model_name=qcfg.get('model_name', 'qwen-plus'))
        engine = UnifiedTextProcessor(llm)
    else:
        # Gemini
        gcfg = cfg.get('gemini_api', {})
        llm = GeminiLLM(api_key_env_var=gcfg.get('api_key_env_var', 'GOOGLE_API_KEY'), 
                        model_name=gcfg.get('model_name', 'gemini-1.5-flash'))
        engine = UnifiedTextProcessor(llm)

    if args.mode == 'evaluate':
        gt_dir = paths.get('ground_truth_dir', 'data/ground_truth')
        # 遍历输出目录下 *_Overall.txt 与 ground_truth 同名.json 比对
        for root, _, files in os.walk(output_dir):
            for f in files:
                if f.endswith('_Overall.txt'):
                    base = f.replace('_Overall.txt', '')
                    pred = os.path.join(root, f)
                    gt_json = os.path.join(gt_dir, f'{base}.json')
                    if os.path.exists(gt_json):
                        m = calculate_metrics(gt_json, pred)
                        print(f"{base}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")
        return

    inputs = iter_inputs(input_dir, args.limit)
    if not inputs:
        print('未找到输入文件（.txt/.pdf）。')
        sys.exit(1)

    for fp in inputs:
        fp_txt = ensure_txt(fp, output_dir)
        # 嵌入相似度筛选（若选择comprehensive或明确要求filter/abstract/summarize前置）
        emb_txt = run_embedding_selection(fp_txt, top_n=int(os.getenv('FCPD_TOP_N', '10')))

        # 将后续输入替换为Embedding筛选结果
        use_txt = emb_txt if os.path.exists(emb_txt) else fp_txt

        res = engine.process_text_file_comprehensive(use_txt, mode=args.mode)
        for k, v in res.items():
            print(f"✓ {k}: {v}")

        # 归档到 engine 专属目录
        base_name = os.path.splitext(os.path.basename(fp_txt))[0]
        subdir = 'gemini' if engine_choice == 'gemini' else 'local llm'
        dest_dir = os.path.join(output_dir, subdir, base_name)
        os.makedirs(dest_dir, exist_ok=True)

        # 源目录（同一目录内可能有多种输出）
        src_dir = os.path.dirname(use_txt)

        # 必备：源txt
        try:
            shutil.copy2(use_txt, os.path.join(dest_dir, os.path.basename(use_txt)))
        except Exception:
            pass

        # 已知产物：从res拷贝
        for v in res.values():
            try:
                if v and os.path.exists(v):
                    shutil.copy2(v, os.path.join(dest_dir, os.path.basename(v)))
            except Exception:
                pass

        # 归档嵌入文件
        try:
            if emb_txt and os.path.exists(emb_txt):
                shutil.copy2(emb_txt, os.path.join(dest_dir, os.path.basename(emb_txt)))
        except Exception:
            pass

    # 清理 data/output 目录（若仍存在）
    legacy_output = os.path.join(output_dir, 'output')
    if os.path.isdir(legacy_output):
        try:
            shutil.rmtree(legacy_output)
            print('🧹 已清理 data/output')
        except Exception as e:
            print(f'⚠️ 清理 data/output 失败: {e}')

    print('✅ 完成')


if __name__ == '__main__':
    main()
