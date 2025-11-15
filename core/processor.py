import os
import pandas as pd
from typing import Dict, Any

from .models.base_llm import BaseLLM


class UnifiedTextProcessor:
    def __init__(self, llm: BaseLLM) -> None:
        self.llm = llm
        self.summary_generation_kwargs = {
            "temperature": 0.0,
            "top_p": 0.6,
        }

    def _create_prompt(self, system: str, user: str, context: str = "") -> str:
        parts = []
        if system:
            parts.append(system)
        if context:
            parts.append(f"Context\n{context}")
        parts.append(f"Task\n{user}")
        return "\n\n".join(parts)

    def save_df_to_text(self, df: pd.DataFrame, file_path: str, content_column: str = 'content') -> None:
        with open(file_path, 'w', encoding='utf-8') as f:
            for _, row in df.iterrows():
                f.write(str(row.get(content_column, "")) + "\n\n")

    def filter_content(self, df: pd.DataFrame) -> pd.DataFrame:
        user_question = (
            "Is the paragraph about flow chemistry/process development with concrete experimental details? "
            "Answer strictly with 'Yes' or 'No'."
        )
        system_prompt = (
            "You are an expert assistant for scientific literature mining. "
            "Classify paragraphs as Yes/No based on whether they contain concrete experimental details."
        )
        classifications = []
        for _, row in df.iterrows():
            content = str(row['content'])
            kw = [
                "flow chemistry","continuous flow","residence time","flow rate","mL/min","µL/min","ul/min",
                "reactor","tubular","coil","microreactor","inner diameter","i.d.","mm","μm",
                "temperature","°c","selectivity","conversion","yield","bpr","bar","back pressure","min","pressure"
            ]
            if any(k in content.lower() for k in kw):
                classifications.append('Yes')
                continue
            prompt = self._create_prompt(system_prompt, user_question, content)
            resp = self.llm.generate(prompt, max_tokens=5, temp=0.0) or ''
            classifications.append('Yes' if resp.strip().lower().startswith('yes') else 'No')
        df = df.copy()
        df['classification'] = classifications
        return df[df['classification'] == 'Yes'].copy()

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
            content = str(row['content'])
            prompt = self._create_prompt(system_prompt, user_prompt, content)
            text = self.llm.generate(prompt, max_tokens=300, temp=0.0) or ''
            abstracts.append(text.strip() or content[:400])
        out = df.copy()
        out['abstract'] = abstracts
        return out

    def _sanitize_json_text(self, text: str) -> str:
        import re
        s = text or ""
        fb, lb = s.find('{'), s.rfind('}')
        if fb != -1 and lb != -1 and lb > fb:
            s = s[fb:lb+1]
        s = re.sub(r"//.*?(?=\n|$)", "", s)
        s = re.sub(r"/\*[\s\S]*?\*/", "", s)
        s = re.sub(r",\s*(\}|\])", r"\1", s)
        s = s.strip()
        return s

    def _keep_best_only(self, json_text: str) -> str:
        """将可能包含多组结果的JSON裁剪为单一最优结果，遵循旧项目择优规则。"""
        import json
        try:
            obj = json.loads(json_text)
        except Exception:
            return json_text
        rs = obj.get("reaction_summary", {})
        # 1) 产品：按 yield_optimal 最大保留一条
        prods = rs.get("products", []) or []
        best_prod = None
        best_yield = None
        for p in prods:
            # 兼容字符串列表和字典列表
            if isinstance(p, str):
                continue
            if isinstance(p, dict):
                y = p.get("yield_optimal")
                if isinstance(y, (int, float)):
                    if best_yield is None or y > best_yield:
                        best_yield = y
                        best_prod = p
        if best_prod is not None:
            rs["products"] = [best_prod]
            met = rs.get("metrics", {}) or {}
            met["yield"] = best_yield
            rs["metrics"] = met
        else:
            # 如果没有产品yield，尝试用metrics选择（yield或conversion最大）
            met = rs.get("metrics", {}) or {}
            # 不裁剪条件，仅保证metrics单一一致
            rs["metrics"] = met
        obj["reaction_summary"] = rs
        try:
            return json.dumps(obj, ensure_ascii=False)
        except Exception:
            return json_text

    def summarize_parameters(self, df) -> pd.DataFrame:
        # 旧项目风格的严格JSON提示词（含示例IO）
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
            "    {\"type\":\"flow_rate_reactant_B\",\"value\":\"...\"},"
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
            "  \"metrics\": {\"conversion\": null, \"yield\": 82, \"selectivity\": 55, \"unit\": \"%\"}"
            "}}"
        )
        summarized = []
        for _, row in df.iterrows():
            content = str(row['content' if 'abstract' not in df.columns else 'abstract'])
            prompt = self._create_prompt(system_prompt, user_prompt, content)
            # 追加严格择优规则，确保仅产出单一条件集
            prompt += (
                "\nRules:\n"
                "- For CONDITIONS and METRICS: choose the OPTIMAL set (highest yield/conversion).\n"
                "- For reaction_type, reactants, products, reactor: use the most informative/complete data (not necessarily from the optimal condition).\n"
                "- If multiple conditions appear, output only ONE optimal condition set.\n"
                "- Use null for unknown fields.\n"
            )
            raw = (self.llm.generate(prompt, max_tokens=300, **self.summary_generation_kwargs) or '').strip()
            start, end = raw.find('{'), raw.rfind('}')
            if start != -1 and end != -1 and end > start:
                raw = raw[start:end+1]
            txt = self._sanitize_json_text(raw)
            txt = self._keep_best_only(txt)
            # 若清洗后仍非JSON，尝试一次降参重试
            try:
                import json as _json
                _json.loads(txt)
                summarized.append(txt)
            except Exception:
                raw2 = (self.llm.generate(prompt, max_tokens=240, **self.summary_generation_kwargs) or '').strip()
                s2, e2 = raw2.find('{'), raw2.rfind('}')
                if s2 != -1 and e2 != -1 and e2 > s2:
                    raw2 = raw2[s2:e2+1]
                txt2 = self._sanitize_json_text(raw2)
                summarized.append(self._keep_best_only(txt2))
        out = df.copy()
        out['summarized'] = summarized
        return out

    def summarize_document_overall(self, df_abstract: pd.DataFrame) -> str:
        import json, re
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
        raw = (self.llm.generate(prompt, max_tokens=1200, **self.summary_generation_kwargs) or "").strip()
        raw = re.sub(r'```json\s*', '', raw)
        raw = re.sub(r'```\s*', '', raw)
        s, e = raw.find("{"), raw.rfind("}")
        if s != -1 and e != -1 and e > s:
            raw = raw[s:e+1]
        cleaned = self._sanitize_json_text(raw)
        # 去除对话/说明文本行
        lines_clean = []
        for ln in cleaned.splitlines():
            low = ln.lower()
            if any(k in low for k in ['note:', 'please', 'let me', 'here', 'option', 'would you', 'i added', 'if you', 'updated json']):
                continue
            lines_clean.append(ln)
        cleaned = "\n".join(lines_clean).strip()
        # 再次仅保留首个完整{...}
        s, e = cleaned.find("{"), cleaned.rfind("}")
        if s != -1 and e != -1 and e > s:
            cleaned = cleaned[s:e+1]
        cleaned = self._keep_best_only(cleaned)
        try:
            json.loads(cleaned)
            return cleaned
        except Exception:
            return cleaned if cleaned else raw

    def _extract_influence_candidates(self, df: pd.DataFrame) -> list:
        import re, os
        content_col = 'abstract' if 'abstract' in df.columns else 'content'
        candidates = []
        topK = int(os.getenv('FCPD_IMPACT_TOPK', '20'))
        for idx, row in df.iterrows():
            p = str(row[content_col])
            if not p or len(p.strip()) < 30:
                continue
            score = 0
            pl = p.lower()
            # 更宽松的候选识别
            if any(k in pl for k in ['effect', 'influence', 'impact', 'affect', 'result', 'increase', 'decrease', 'improve', 'enhance', 'reduce']):
                score += 15
            if any(k in pl for k in ['selectivity', 'conversion', 'yield', 'performance', 'efficiency']):
                score += 10
            if re.search(r'\b\d+\.?\d*\s*(%|°c|k|ml/min|min|bar|mpa)\b', p, re.IGNORECASE):
                score += 8
            # 因果动词bonus
            if any(v in pl for v in ['higher', 'lower', 'longer', 'shorter', 'faster', 'slower']):
                score += 5
            if score >= 5:
                candidates.append({'index': idx, 'text': p, 'score': score})
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
            return self._to_markdown_impact3([])
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
        raw = self.llm.generate(prompt, max_tokens=700, temp=0.3) or ""
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
        print(f"🔍 处理文件: {os.path.basename(file_path)}")
        
        # 🚀 在线LLM快速模式：直接处理全文（仅Overall+Impact）
        if mode == 'fast':
            return self._process_fast_direct(file_path)
        
        # 标准5步流程（本地LLM和在线LLM都支持）
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

            # Overall JSON（基于抽象优先）
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
    
    def _process_fast_direct(self, file_path: str) -> Dict[str, Any]:
        """在线LLM快速模式：直接从全文提取Overall+Impact，跳过中间步骤"""
        import re, json
        print("  ⚡ 在线LLM快速模式：直接处理全文PDF→JSON+Impact")
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            full_text = f.read()
        
        # 限制文本长度（在线LLM上下文窗口大）
        if len(full_text) > 80000:
            full_text = full_text[:80000]
        
        outputs = {}
        
        # 1. 一次性提取Overall JSON
        print("  📊 提取Overall JSON（从全文）...")
        system = "You are an expert in flow chemistry literature analysis."
        user = f"""Extract the OPTIMAL flow chemistry parameters from this paper as ONE JSON object.

Example output format:
{{"reaction_summary":{{"reaction_type":"hydrogenation","reactants":["furfural","H2","Pd/C catalyst"],"products":["furfuryl alcohol"],"conditions":[{{"type":"temperature","value":"80 °C"}},{{"type":"residence_time","value":"5 min"}},{{"type":"pressure","value":"2 MPa"}}],"reactor":{{"type":"packed bed","inner_diameter":"5 mm"}},"metrics":{{"conversion":95.2,"yield":89.5,"selectivity":94.1,"unit":"%"}}}}}}

Rules:
- Extract reaction_type, all reactants (with catalysts), specific product names
- Extract OPTIMAL conditions (highest yield/conversion reported in paper)
- Include reactor type and dimensions
- Metrics as numbers (conversion, yield, selectivity)
- Use null for unknown fields

Paper full text:
{full_text}

Output ONLY valid JSON:"""
        
        prompt = self._create_prompt(system, user, "")
        raw = (self.llm.generate(prompt, max_tokens=1000, temp=0.1) or "").strip()
        
        # 清洗JSON
        raw = re.sub(r'```json\s*', '', raw, flags=re.IGNORECASE)
        raw = re.sub(r'```\s*', '', raw)
        s, e = raw.find('{'), raw.rfind('}')
        if s != -1 and e != -1 and e > s:
            raw = raw[s:e+1]
        
        cleaned = self._sanitize_json_text(raw)
        cleaned = self._keep_best_only(cleaned)
        
        overall_file = file_path.replace('.txt', '_Overall.txt')
        with open(overall_file, 'w', encoding='utf-8') as f:
            f.write(cleaned)
        outputs['summarized_overall'] = overall_file
        print(f"  ✅ Overall → {overall_file}")
        
        # 2. 一次性提取Impact
        print("  📊 提取影响因素（从全文）...")
        impact_sys = "Extract cause-effect relationships from chemical papers."
        impact_user = f"""Extract ALL Factor-Metric-Direction relationships from this flow chemistry paper.

Example:
Input: "A longer residence time results in higher conversion. Selectivity remains unchanged."
Output:
residence_time | conversion | increase
residence_time | selectivity | unchanged

Rules:
- Format: Factor | Metric | Direction (one per line)
- Direction: increase, decrease, or unchanged
- No table headers

Paper text (first 40K chars):
{full_text[:40000]}

Output relationships:"""
        
        prompt_impact = self._create_prompt(impact_sys, impact_user, "")
        raw_impact = (self.llm.generate(prompt_impact, max_tokens=800, temp=0.2) or "").strip()
        
        # 解析Impact
        items = []
        for line in raw_impact.splitlines():
            if '|' not in line:
                continue
            low = line.lower()
            if 'factor' in low and 'metric' in low:
                continue
            parts = [p.strip() for p in line.split('|')]
            if len(parts) < 3 or not parts[0] or not parts[1] or parts[0] == '-':
                continue
            direction = parts[2].lower()
            if 'increase' in direction or 'higher' in direction or 'improve' in direction:
                direction = 'increase'
            elif 'decrease' in direction or 'lower' in direction or 'reduce' in direction:
                direction = 'decrease'
            elif 'unchange' in direction or 'no effect' in direction:
                direction = 'unchanged'
            else:
                direction = '-'
            items.append({'factor': parts[0], 'metric': parts[1], 'direction': direction})
        
        impact_md = self._to_markdown_impact(items)
        impact_file = file_path.replace('.txt', '_Impact_Analysis.txt')
        with open(impact_file, 'w', encoding='utf-8') as f:
            f.write("# Influence Factor Summary\n\n")
            f.write(impact_md)
        outputs['impact_analysis'] = impact_file
        print(f"  ✅ Impact → {impact_file}")
        print(f"  ⚡ 快速模式完成（2次API调用）")
        
        return outputs


