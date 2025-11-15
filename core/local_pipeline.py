from __future__ import annotations

import os
from typing import Dict, Any

import pandas as pd
from gpt4all import GPT4All


class LocalPipeline:
    def __init__(self, model_name: str | None = None, model_path: str = 'models/', *, filter_model: str | None = None, abstract_model: str | None = None, summarize_model: str | None = None, finetuned_trigger_name: str | None = None) -> None:
        abs_model_path = os.path.abspath(model_path)
        # 单模型或分阶段模型装配
        self.model_path = abs_model_path
        self.model_single = None
        if model_name:
            self.model_single = GPT4All(model_name, model_path=abs_model_path, allow_download=False)
        self.model_filter_name = filter_model
        self.model_abstract_name = abstract_model
        self.model_summarize_name = summarize_model
        self.finetuned_trigger_name = finetuned_trigger_name or 'My_Finetuned_Model'

    def _create_prompt(self, system: str, user: str, context: str = "") -> str:
        """创建普通格式的 prompt"""
        parts = []
        if system:
            parts.append(system)
        if context:
            parts.append(f"Context\n{context}")
        parts.append(f"Task\n{user}")
        return "\n\n".join(parts)
    def _create_llama31_chat_prompt(self, system: str, user: str, context: str = "") -> str:
        """创建 Llama 3.1 chat template 格式的 prompt（用于微调模型）"""
        user_content = ""
        if context:
            user_content += f"Context\n{context}\n\n"
        user_content += f"Task\n{user}"
        
        # Llama 3.1 chat template 格式
        prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        return prompt

    def _is_finetuned_model(self, model_name: str) -> bool:
        """判断是否是微调模型"""
        return self.finetuned_trigger_name.lower() in model_name.lower()

    def _get_chat_prompt(self, system: str, user: str, context: str = "", stage: str = "filter") -> str:
        """根据模型类型选择 prompt 格式"""
        # 判断当前阶段的模型是否是微调模型
        if stage == 'summarize' and self.model_summarize_name:
            if self._is_finetuned_model(self.model_summarize_name):
                return self._create_llama31_chat_prompt(system, user, context)
        return self._create_prompt(system, user, context)

    

    def _clip(self, text: str, max_chars: int) -> str:
        if not text:
            return text
        if len(text) <= max_chars:
            return text
        return text[:max_chars]

    def _safe_generate(self, model: GPT4All, prompt: str, *, max_tokens: int, temp: float = 0.0) -> str:
        try:
            return (model.generate(prompt=prompt, max_tokens=max_tokens, temp=temp) or '').strip()
        except Exception:
            # 二次降载：减少prompt与max_tokens后重试
            clipped = self._clip(prompt, 2000)
            try:
                return (model.generate(prompt=clipped, max_tokens=max_tokens - 40, temp=temp) or '').strip()
            except Exception:
                # 最终兜底
                clipped = self._clip(prompt, 1500)
                return (model.generate(prompt=clipped, max_tokens=max_tokens - 80, temp=temp) or '').strip()

    def save_df_to_text(self, df: pd.DataFrame, file_path: str, content_column: str = 'content') -> None:
        with open(file_path, 'w', encoding='utf-8') as f:
            for _, row in df.iterrows():
                f.write(str(row.get(content_column, "")) + "\n\n")

    def filter_content(self, df: pd.DataFrame) -> pd.DataFrame:
        user_question = (
            "Does the paragraph contain experimental details about flow-chemistry/process development? "
            "Answer strictly with 'Yes' or 'No'."
        )
        system_prompt = (
            "You are an expert assistant for scientific literature mining. "
            "Classify paragraphs as Yes/No based on whether they contain concrete experimental details."
        )
        results = []
        for _, row in df.iterrows():
            content = self._clip(str(row['content']), 1200)
            kw = [
                "flow chemistry","continuous flow","residence time","flow rate","mL/min","µL/min","ul/min",
                "reactor","tubular","coil","microreactor","inner diameter","i.d.","mm","μm",
                "temperature","°c","selectivity","conversion","yield","bpr","bar","back pressure","min","pressure"
            ]
            if any(k in content.lower() for k in kw):
                results.append('Yes')
                continue
            prompt = self._create_prompt(system_prompt, user_question, content)
            model = self._get_stage_model('filter')
            resp = self._safe_generate(model, prompt, max_tokens=5, temp=0.0)
            results.append('Yes' if resp.strip().lower().startswith('yes') else 'No')
        out = df.copy()
        out['classification'] = results
        return out[out['classification'] == 'Yes'].copy()

    def abstract_text(self, df: pd.DataFrame) -> pd.DataFrame:
        user_prompt = (
            "Summarize the paragraph focusing on flow-chemistry process development. "
            "Highlight: reaction type, key reactants/solvent/catalyst, products, reactor details (type/ID), "
            "critical conditions (flow rate(s), residence time, temperature, pressure), and outcomes (conversion/yield/selectivity). "
            "Be concise, faithful to text, no speculation."
        )
        system_prompt = (
            "You are an expert assistant for scientific literature mining. Return concise, faithful summaries."
        )
        abstracts = []
        for _, row in df.iterrows():
            content = self._clip(str(row['content']), 2000)
            prompt = self._create_prompt(system_prompt, user_prompt, content)
            model = self._get_stage_model('abstract')
            text = self._safe_generate(model, prompt, max_tokens=280, temp=0.0)
            abstracts.append(text.strip() or content[:400])
        out = df.copy()
        out['abstract'] = abstracts
        return out

    def _sanitize_json_text(self, text: str) -> str:
        import re
        s = text or ""
        # 1) 去除 markdown 围栏
        s = re.sub(r"```json\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"```\s*", "", s)
        # 2) 仅保留最大完整 { ... } 块
        brace_count = 0
        start = -1
        candidates = []
        for i, ch in enumerate(s):
            if ch == '{':
                if brace_count == 0:
                    start = i
                brace_count += 1
            elif ch == '}':
                if brace_count > 0:
                    brace_count -= 1
                    if brace_count == 0 and start != -1:
                        candidates.append(s[start:i+1])
                        start = -1
        if candidates:
            s = max(candidates, key=len)
        # 3) 去注释与尾逗号
        s = re.sub(r"//.*?(?=\n|$)", "", s)
        s = re.sub(r"/\*[\s\S]*?\*/", "", s)
        s = re.sub(r",\s*(\}|\])", r"\1", s)
        # 4) 数值化（将 "82" → 82，仅限纯数字的值）
        s = re.sub(r':\s*"(-?\d+\.?\d*)"\s*([,\}])', r': \1\2', s)
        return s.strip()

    def _keep_best_only(self, json_text: str) -> str:
        import json
        try:
            obj = json.loads(json_text)
        except Exception:
            return json_text
        rs = obj.get("reaction_summary", {})
        prods = rs.get("products", []) or []
        
        # 兼容products为字符串列表或字典列表
        if not prods:
            return json_text
        
        best_prod = None
        best_yield = None
        
        for p in prods:
            # 如果是字符串，跳过（不处理）
            if isinstance(p, str):
                continue
            # 如果是字典，提取yield_optimal
            if isinstance(p, dict):
                y = p.get("yield_optimal")
                if isinstance(y, (int, float)):
                    if best_yield is None or y > best_yield:
                        best_yield = y
                        best_prod = p
        
        # 只有找到有效的best_prod才进行替换
        if best_prod is not None:
            rs["products"] = [best_prod]
            met = rs.get("metrics", {}) or {}
            met["yield"] = best_yield
            rs["metrics"] = met
            obj["reaction_summary"] = rs
        
        try:
            return json.dumps(obj, ensure_ascii=False)
        except Exception:
            return json_text

    def summarize_parameters(self, df: pd.DataFrame) -> pd.DataFrame:
        system_prompt = (
            "You output ONLY valid JSON. No explanations, no markdown, no comments."
        )
        user_prompt = (
            "Only use the provided paragraph; do not infer across other paragraphs.\n"
            "If a field is not explicitly stated, use null. Use original units when present; otherwise normalize: "
            "temperature in °C, residence_time in min, flow_rate in mL/min, inner_diameter in mm.\n"
            "Output ONLY the following JSON object (no extra text):\n"
            "{ \"reaction_summary\": {"
            "  \"reaction_type\":\"...\"," 
            "  \"reactants\":[{\"name\":\"...\",\"role\":\"reactant|catalyst|solvent\"}],"
            "  \"products\":[{\"name\":\"...\",\"yield_optimal\":95,\"unit\":\"%\"}],"
            "  \"conditions\":["
            "    {\"type\":\"temperature\",\"value\":\"...\"},"
            "    {\"type\":\"residence_time\",\"value\":\"...\"},"
            "    {\"type\":\"flow_rate_reactant_A\",\"value\":\"...\"},"
            "    {\"type\":\"flow_rate_total\",\"value\":\"...\"},"
            "    {\"type\":\"pressure\",\"value\":\"...\"}"
            "  ],"
            "  \"reactor\":{\"type\":\"...\",\"inner_diameter\":\"...\"},"
            "  \"metrics\":{\"conversion\":...,\"yield\":...,\"selectivity\":...,\"unit\":\"%\"}"
            "}}\n"
            "Example input: \"Flow rate 0.1 mL/min, T=80 °C in a 0.5 mm coil; yield 82%.\"\n"
            "Example output: { \"reaction_summary\": {"
            "  \"reaction_type\": null, \"reactants\": [],"
            "  \"products\": [{\"name\": null, \"yield_optimal\": 82, \"unit\": \"%\"}],"
            "  \"conditions\": [ {\"type\":\"temperature\",\"value\":\"80 °C\"}, {\"type\":\"flow_rate_total\",\"value\":\"0.1 mL/min\"} ],"
            "  \"reactor\": {\"type\":\"coil\", \"inner_diameter\":\"0.5 mm\"},"
            "  \"metrics\": {\"conversion\": null, \"yield\": 82, \"selectivity\": null, \"unit\": \"%\"}"
            "}}\n"
            "Rules:\n"
            "- For CONDITIONS and METRICS: choose the OPTIMAL set (highest yield/conversion).\n"
            "- For reaction_type, reactants, products, reactor: use the most informative/complete data (not necessarily from the optimal condition).\n"
            "- If multiple conditions appear, output only ONE optimal condition set.\n"
            "- Use null for unknown fields.\n"
        )
        summarized = []
        for _, row in df.iterrows():
            content = str(row['content' if 'abstract' not in df.columns else 'abstract'])
            content = self._clip(content, 2200)
            
            # 使用新的 _get_chat_prompt 方法，自动选择格式
            base_prompt = self._get_chat_prompt(system_prompt, user_prompt, content, stage='summarize')
            
            # 追加择优规则（但不要追加到 chat template 的 assistant 标记之后）
            # 需要把 Rules 放在 Task 的 user_prompt 中
            # ... 修改 user_prompt 包含 Rules ...
            model = self._get_stage_model('summarize')
            raw = self._safe_generate(model, base_prompt, max_tokens=512, temp=0.0)
            
            # 去围栏并抽取花括号块
            
            import re as _re
            raw = _re.sub(r"```json\s*", "", raw, flags=_re.IGNORECASE)
            raw = _re.sub(r"```\s*", "", raw)
            start, end = raw.find('{'), raw.rfind('}')
            if start != -1 and end != -1 and end > start:
                raw = raw[start:end+1]
            txt = self._sanitize_json_text(raw)
            txt = self._keep_best_only(txt)
            summarized.append(txt)
        out = df.copy()
        out['summarized'] = summarized
        return out

    def summarize_document_overall(self, df_abstract: pd.DataFrame) -> str:
        """参考旧项目：将抽象/内容汇总为单一JSON（择优合并）。"""
        import re, json
        col = 'abstract' if 'abstract' in df_abstract.columns else 'content'
        texts = [t for t in df_abstract[col].fillna("").tolist() if t.strip()]
        combined = "\n\n".join(texts)
        if len(combined) > 12000:
            combined = combined[:12000]

        system_prompt = "You output ONLY valid JSON. No explanations."
        user_prompt = (
            "Extract the OPTIMAL condition set from abstracts. Output ONE JSON:\n"
            '{"reaction_summary":{"reaction_type":"hydrogenation","reactants":["furfural","H2","Pd/C catalyst"],'
            '"products":["furfuryl alcohol"],'
            '"conditions":[{"type":"temperature","value":"80 °C"},{"type":"residence_time","value":"5 min"},{"type":"pressure","value":"2 MPa"}],'
            '"reactor":{"type":"packed bed","inner_diameter":"5 mm"},'
            '"metrics":{"conversion":95.2,"yield":89.5,"selectivity":94.1,"unit":"%"}}}\n'
            "Choose best yield/conversion. Use null if unknown. Numbers for metrics.\n"
        )
        prompt = self._create_prompt(system_prompt, user_prompt, combined)
        raw = self._safe_generate(self._get_stage_model('summarize'), prompt, max_tokens=900, temp=0.05)
        # 去围栏并抽取最大花括号块
        raw = re.sub(r"```json\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"```\s*", "", raw)
        s, e = raw.find("{"), raw.rfind("}")
        if s != -1 and e != -1 and e > s:
            raw = raw[s:e+1]
        cleaned = self._sanitize_json_text(raw)
        cleaned = self._keep_best_only(cleaned)
        try:
            json.loads(cleaned)
            return cleaned
        except Exception:
            return raw

    def _extract_influence_candidates(self, df: pd.DataFrame) -> list:
        import re, os
        content_col = 'abstract' if 'abstract' in df.columns else 'content'
        candidates = []
        topK = int(os.getenv('FCPD_IMPACT_TOPK', '12'))
        
        causal_verbs = [
            'increase', 'decrease', 'improve', 'enhance', 'reduce', 'affect',
            'influence', 'impact', 'promote', 'inhibit', 'facilitate', 'optimize',
            'control', 'determine', 'depend', 'vary', 'change', 'modulate'
        ]
        metric_keywords = [
            'conversion', 'yield', 'selectivity', 'purity', 'efficiency',
            'product distribution', 'heat transfer', 'mixing'
        ]
        
        for idx, row in df.iterrows():
            paragraph = str(row[content_col])
            if not paragraph or len(paragraph.strip()) < 30:
                continue
            score = 0
            para_lower = paragraph.lower()
            
            # Results/Discussion sections
            is_results = any(sec in para_lower for sec in ['result', 'discussion', 'finding', 'observation'])
            if is_results:
                has_metric = any(kw in para_lower for kw in metric_keywords)
                if has_metric:
                    score += 25
                else:
                    score += 10
            
            # 因果动词
            causal_count = sum(1 for verb in causal_verbs if verb in para_lower)
            if causal_count > 0:
                has_metric = any(m in para_lower for m in metric_keywords)
                if has_metric:
                    score += 8 * min(causal_count, 3)
                else:
                    score += 3 * min(causal_count, 2)
            
            # 定量数据
            quant_matches = re.findall(r'\b\d+\.?\d*\s*(?:%|K|°C|°F|mL|L|min|h|MPa|bar|M|mol|g)\b', paragraph)
            if len(quant_matches) >= 2:
                score += 10
            
            if score >= 3:  # 进一步放宽阈值，确保有足够候选
                candidates.append({'index': idx, 'text': paragraph, 'score': score})
        
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates[:topK]

    def _to_markdown_impact(self, items: list) -> str:
        if not items:
            return "| Factor | Metric | Direction |\n|--------|--------|-----------|\n| None | - | - |"
        lines = [
            "| Factor | Metric | Direction |",
            "|--------|--------|-----------|",
        ]
        for it in items:
            factor = (it.get('factor','-') or '-').replace('|','\\|')
            metric = (it.get('metric','-') or '-').replace('|','\\|')
            direction = (it.get('direction','') or '').lower()
            if direction not in ['increase','decrease','unchanged']:
                direction = '-'
            lines.append(f"| {factor} | {metric} | {direction} |")
        return "\n".join(lines)

    def extract_influence_factors_with_llm(self, df: pd.DataFrame) -> str:
        cands = self._extract_influence_candidates(df)
        print(f"  🐛 Impact候选段落数: {len(cands)}")
        if not cands:
            return self._to_markdown_impact([])
        joined = "\n\n".join(c['text'][:800] for c in cands[:min(len(cands), 15)])
        print(f"  🐛 拼接文本长度: {len(joined)} 字符")

        # Few-shot 简化但完整的案例
        system_prompt = "Extract cause-effect relationships from chemical paragraphs."
        user_prompt = (
            "Example paragraph:\n"
            "Residence time is an important parameter. A longer residence time will result in "
            "a higher conversion of TFMB. The product selectivity nearly remains unchanged.\n\n"
            "Example output:\n"
            "residence_time | conversion of TFMB | increase\n"
            "residence_time | product selectivity | unchanged\n\n"
            "Extract Factor-Metric-Direction from the paragraphs below.\n"
            "Format: Factor | Metric | Direction (one per line, no table headers)\n"
            "Direction: increase, decrease, or unchanged\n\n"
            "Paragraphs:\n"
        )
        prompt = self._create_prompt(system_prompt, user_prompt, joined)
        model = self._get_stage_model('summarize')
        raw = self._safe_generate(model, prompt, max_tokens=700, temp=0.3)
        print(f"  🐛 LLM原始输出前500字符: {raw[:500]}")

        # 解析为三列（兼容2列和3列格式）
        items = []
        for line in raw.splitlines():
            if '|' not in line:
                continue
            # 跳过表头行
            low = line.lower()
            if 'factor' in low and 'metric' in low:
                continue
            parts = [p.strip() for p in line.split('|')]
            if len(parts) < 2:  # 改为至少2列即可
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
                    metric = parts[1].split()[0]  # 取第一个词作为metric
                elif 'decrease' in metric_with_dir or 'lower' in metric_with_dir:
                    direction = 'decrease'
                    metric = parts[1].split()[0]
                elif 'unchange' in metric_with_dir:
                    direction = 'unchanged'
                    metric = parts[1].split()[0]
                else:
                    direction = '-'
                    metric = parts[1]
            else:
                # 三列格式：Factor | Metric | Direction
                factor = parts[0]
                metric = parts[1]
                direction = parts[2].lower()
                if 'increase' in direction or 'higher' in direction or 'improve' in direction or 'enhance' in direction:
                    direction = 'increase'
                elif 'decrease' in direction or 'lower' in direction or 'reduce' in direction or 'inhibit' in direction:
                    direction = 'decrease'
                elif 'unchange' in direction or 'unchanged' in direction or 'no effect' in direction:
                    direction = 'unchanged'
                else:
                    direction = ''
            
            items.append({'factor': factor, 'metric': metric, 'direction': direction})
        return self._to_markdown_impact(items)

    def process_text_file_comprehensive(self, file_path: str, mode: str = 'comprehensive') -> Dict[str, Any]:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        segs, cur = [], []
        for line in lines:
            if line.strip():
                cur.append(line.strip())
            else:
                if cur:
                    segs.append(' '.join(cur))
                    cur = []
        if cur:
            segs.append(' '.join(cur))
        df = pd.DataFrame(segs, columns=['content'])

        outputs: Dict[str, Any] = {}
        if mode in ['filter', 'comprehensive']:
            df_filtered = self.filter_content(df)
            filter_file = file_path.replace('.txt', '_Filtered.txt')
            self.save_df_to_text(df_filtered, filter_file)
            outputs['filter'] = filter_file
        if mode in ['abstract', 'comprehensive']:
            df_abstract = self.abstract_text(df_filtered if 'df_filtered' in locals() else df)
            abstract_file = file_path.replace('.txt', '_Abstract.txt')
            self.save_df_to_text(df_abstract, abstract_file, 'abstract')
            outputs['abstract'] = abstract_file
        if mode in ['summarize', 'comprehensive']:
            df_input = df_abstract if 'df_abstract' in locals() else (df_filtered if 'df_filtered' in locals() else df)
            df_summarized = self.summarize_parameters(df_input)
            summarize_file = file_path.replace('.txt', '_Summarized.txt')
            self.save_df_to_text(df_summarized, summarize_file, 'summarized')
            outputs['summarized'] = summarize_file
            # Overall（基于抽象优先）
            overall_input = df_abstract if 'df_abstract' in locals() else df_input
            overall_json = self.summarize_document_overall(overall_input)
            overall_file = file_path.replace('.txt', '_Overall.txt')
            with open(overall_file, 'w', encoding='utf-8') as f:
                f.write(overall_json)
            outputs['summarized_overall'] = overall_file

            # 影响因素（简化三列）
            try:
                impact_md = self.extract_influence_factors_with_llm(overall_input)
                impact_file = file_path.replace('.txt', '_Impact_Analysis.txt')
                with open(impact_file, 'w', encoding='utf-8') as f:
                    f.write("# Influence Factor Summary\n\n")
                    f.write(impact_md)
                outputs['impact_analysis'] = impact_file
            except Exception:
                pass
        return outputs

    def _get_stage_model(self, stage: str) -> GPT4All:
        """按阶段返回对应模型；若未配置分阶段，则回退到单模型。"""
        if self.model_single is not None:
            return self.model_single
        name = None
        if stage == 'filter':
            name = self.model_filter_name
        elif stage == 'abstract':
            name = self.model_abstract_name
        elif stage == 'summarize':
            name = self.model_summarize_name
        if not name:
            # 兜底：使用摘要模型或任一可用
            name = self.model_abstract_name or self.model_summarize_name or self.model_filter_name
        return GPT4All(name, model_path=self.model_path, allow_download=False)