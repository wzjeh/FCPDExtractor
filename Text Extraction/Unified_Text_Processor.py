from gpt4all import GPT4All
import pandas as pd
import os
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class UnifiedTextProcessor:
    """
    统一的文本处理类，整合所有文本处理功能
    """
    
    def __init__(self, model_name='nous-hermes-llama2-13b.Q4_0.gguf', model_path='models/', strict=False):
        self.model_name = model_name
        self.model_path = model_path
        self.strict = strict  # 严格使用指定模型（失败不回退）
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.model = self.load_llm_model() # 在初始化时只加载一次
        # 角色扮演，系统指令扮演专家助理角色
        self.system_prompt = (
            "You are an expert assistant for scientific literature mining. "
            "Your task is to follow the user's instructions precisely to extract structured data from scientific texts."
        )
        
    def _create_prompt(self, user_prompt, context=""):
        """
        一个辅助函数，用于创建带有系统指令的完整Prompt。
        """
        # 使用分隔符让结构更清晰
        return (
            f"{self.system_prompt}\n\n"
            f"### Paragraph to Analyze ###\n"
            f"{context}\n\n"
            f"### Task ###\n"
            f"{user_prompt}"
        )
    def load_llm_model(self):
        """
        加载LLM模型，支持回退机制
        """
        # 获取绝对路径
        abs_model_path = os.path.abspath(self.model_path)
        print(f"🔍 尝试加载模型，路径: {abs_model_path}")
        # 严格模式：首选本地文件（models 目录）禁止下载；若未找到，再尝试默认缓存目录（仍禁止下载）
        if self.strict:
            strict_name = os.getenv('FCPD_STRICT_MODEL_NAME') or self.model_name
            print(f"🔒 严格模式，目标模型: {strict_name}")
            try:
                # 首选 models 目录下本地文件（不下载）
                model = GPT4All(strict_name, model_path=abs_model_path, allow_download=False)
                print(f"✅ 成功加载(严格, 本地models目录) {strict_name} 模型")
                return model
            except Exception as e:
                print(f"❌ 严格模式本地models目录加载失败: {e}")
                try:
                    # 再尝试默认缓存目录（不下载）
                    model = GPT4All(strict_name, allow_download=False)
                    print(f"✅ 成功加载(严格, 默认缓存目录) {strict_name} 模型")
                    return model
                except Exception as e2:
                    print(f"❌ 严格模式默认缓存目录也失败: {e2}")
                    raise e2

        try:
            # 首先尝试使用绝对路径加载指定模型
            model = GPT4All(self.model_name, model_path=abs_model_path, allow_download=False)
            print(f"✅ 成功加载 {self.model_name} 模型")
            return model
        except Exception as e:
            print(f"❌ 加载 {self.model_name} 失败: {e}")
            print("🔄 尝试使用默认路径...")
            try:
                # 尝试使用默认路径
                model = GPT4All(self.model_name, allow_download=False)
                print(f"✅ 成功加载 {self.model_name} 模型 (默认路径)")
                return model
            except Exception as e2:
                print(f"❌ 默认路径也失败: {e2}")
                print("🔄 尝试使用Meta-Llama-3.1-8B模型...")
                try:
                    # 尝试Meta-Llama模型
                    model = GPT4All('Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf', model_path=abs_model_path, allow_download=False)
                    print("✅ 成功加载 Meta-Llama-3.1-8B 模型 (备选)")
                    return model
                except Exception as e3:
                    print(f"❌ Meta-Llama模型也失败: {e3}")
                    print("🔄 最后回退到 nous-hermes-llama2-13b 模型...")
                    try:
                        # 最后回退到nous-hermes模型
                        model = GPT4All('nous-hermes-llama2-13b.Q4_0.gguf', model_path=abs_model_path, allow_download=False)
                        print("✅ 成功加载 nous-hermes-llama2-13b 模型 (最终回退)")
                        return model
                    except Exception as e4:
                        print(f"❌ 所有模型加载尝试都失败: {e4}")
                        raise e4
    
    # def filter_content_with_llm(self, df):
    #     """
    #     使用LLM过滤内容（替代Filter.py功能）
    #     """
    #     # model = self.load_llm_model()
        
    #     ## 原Filter.py功能
    #     # questions = [
    #     #     "Question: Does this section cover the types of surface chemical reactions or experimental studies on the formation of molecules on surfaces? Answer 'Yes' or 'No'. \nAnswer:"
    #     # ]
    #     questions = [
    #     ("Question: Is this paragraph about flow chemistry or process development, including "
    #      "continuous flow setup, reactor type/ID, flow rates, residence time, temperature, reactant, "
    #      "catalyst, optimization, conversion/yield/selectivity? Answer 'Yes' or 'No'.\nAnswer:")
    #     ]
        
    #     for idx, row in df.iterrows():
    #         content = row['content']
    #         classification = 'No'
            
    #         for question in questions:
    #             prompt = f"{content}\n{question}"
    #             try:
    #                 response = self.model.generate(prompt=prompt, max_tokens=10, temp=0.1)
    #                 if response and response.strip():
    #                     first_word = response.split()[0].replace('.', '').replace(',', '')
    #                     if first_word not in ['No', 'Not']:
    #                         classification = first_word
    #                         break
    #             except Exception as e:
    #                 print(f"Error generating response: {e}")
    #                 continue
            
    #         df.loc[idx, 'classification'] = classification
        
    #     # 过滤掉"No"的段落
    #     condition = (df['classification'] != 'No') & (df['classification'] != 'Not')
    #     df_filtered = df[condition]
        
    #     return df_filtered

    def filter_content_with_llm(self, df):
        """
        使用LLM过滤内容，已使用新的Prompt结构进行优化。
        """
        # 1. 将核心问题定义得更清晰，作为用户指令
        # user_question = (
        #     "Based on the criteria below, does the provided paragraph describe an experimental procedure "
        #     "for flow chemistry or its process development? Answer strictly with 'Yes' or 'No'.\n\n"
        #     "Criteria: The paragraph should mention specific experimental details, for example: "
        #     "continuous flow setup, reactor type/ID, flow rates, residence time, temperature, reactant, "
        #     "catalyst, optimization, or conversion/yield/selectivity.\n\n"
        #     "Answer:"
        # ) # 过于严格了，没结果改成下边的

        user_question = (
            "Does the paragraph contain experimental details about flow-chemistry/process development? "
            "Answer strictly with 'Yes' or 'No'."
        )
        # 导入多线程库
        classifications = []  # 创建一个列表来收集所有分类结果，比逐行修改DataFrame更高效
        
        print("...开始使用LLM进行段落分类...")
        # 2. 遍历DataFrame的每一行
        for index, row in df.iterrows():
            content = row['content']
            content_low = content.lower()
            kw = [
                "flow chemistry","continuous flow","residence time","flow rate","mL/min","µL/min","ul/min",
                "reactor","tubular","coil","microreactor","inner diameter","i.d.","mm","μm",
                "temperature","°c","selectivity","conversion","yield","bpr","bar","back pressure","min","pressure"
            ]
            # 新增：关键词直通，避免过严导致0段落
            if any(k in content_low for k in kw):
                classifications.append('Yes')
                continue
            
            # 3. 使用您的辅助函数创建完整的、带有上下文和系统指令的Prompt
            # 假设 self.system_prompt 和 self._create_prompt 已在类中定义
            full_prompt = self._create_prompt(user_prompt=user_question, context=content)
            
            try:
                # 4. 调用模型生成响应
                # 将temp设为0.0，让模型的回答更具确定性（减少随机性）
                response = self.model.generate(prompt=full_prompt, max_tokens=5, temp=0.0)
                
                # 5. 对响应进行更稳健的解析
                # .strip() 去除首尾空格, .lower() 转为小写, .startswith('yes') 判断是否以'yes'开头
                if response and response.strip().lower().startswith('yes'):
                    classifications.append('Yes')
                else:
                    classifications.append('No')

            except Exception as e:
                print(f"处理第 {index} 行时发生错误: {e}")
                classifications.append('No')  # 如果出错，默认为'No'

        # 6. 一次性将所有分类结果添加到DataFrame中
        df['classification'] = classifications
        
        # 7. 过滤掉 "No" 的段落，并使用 .copy() 避免潜在的警告
        df_filtered = df[df['classification'] == 'Yes'].copy()
        
        print(f"...分类完成，保留 {len(df_filtered)} 个相关段落。")
        return df_filtered
    
    def create_abstract_conclusion_embeddings(self, df):
        """
        创建摘要和结论专用的嵌入（整合Abstract_Conclusion_Embedding.py功能）
        """
        # 定义摘要和结论相关的关键词
        abstract_conclusion_keywords = [
            "conclusion", "abstract", "summary", "findings", "results", 
            "flow chemistry", "continuous flow", "process development", "reactor",
            "flow rate", "residence time", "optimization", "scale-up", "yield",
            "conversion", "selectivity", "catalyst", "temperature", "pressure"
        ]
        
        # 创建参考文本
        reference_text = " ".join(abstract_conclusion_keywords)
        
        # 计算嵌入
        df['content_embedding'] = df['content'].apply(lambda x: self.embedding_model.encode(x, convert_to_tensor=True))
        reference_embedding = self.embedding_model.encode(reference_text, convert_to_tensor=True)
        
        # 计算相似度
        df['similarity'] = df['content_embedding'].apply(
            lambda x: cosine_similarity([x.cpu().numpy()], [reference_embedding.cpu().numpy()])[0][0]
        )
        
        return df
    
    def select_top_paragraphs(self, df, top_n=10):
        """
        选择最相关的段落
        """
        df_sorted = df.sort_values('similarity', ascending=False)
        return df_sorted.head(top_n)
    
    def abstract_text_with_llm(self, df):
        """
        使用LLM进行文本抽象（已优化）
        """
        abstract = []
        
        # 1. 定义针对此任务的用户指令
        user_prompt = (
            "Please summarize the paragraph focusing on flow-chemistry process development. "
            "The summary should highlight: reaction type, reactants/catalyst, products, reactor details, "
            "key conditions (like flow rates, residence time, temperature), "
            "and any reported outcomes (conversion/yield/selectivity). Be concise and faithful to the source text."
        )
        
        for index, row in df.iterrows():
            content = row['content']
            
            # 2. 使用辅助函数构建完整的Prompt
            full_prompt = self._create_prompt(user_prompt=user_prompt, context=content)
            
            try:
                # 3. 使用 self.model 进行调用
                # abstract_text = self.model.generate(prompt=full_prompt, max_tokens=250, temp=0.0, top_p=0.6)
                # if not abstract_text:
                #     # 兜底：避免空摘要，保留上下文的一个精简片段
                #     abstract_text = content[:400]
                abstract_text = self.model.generate(prompt=full_prompt, max_tokens=300, temp=0.0, top_p=0.5)
                abstract_text = (abstract_text or "").strip()
                if not abstract_text:
                    # 兜底：用原段落截断，保证后续文件非空
                    abstract_text = content[:400]

                print(f"Abstract {index+1}/{len(df)}:")
                print(abstract_text)
                abstract.append(abstract_text)
            except Exception as e:
                print(f"Error generating abstract for row {index}: {e}")
                abstract.append(f"Error: {e}")
        
        df['abstract'] = pd.Series(abstract, index=df.index) # 确保索引对齐
        return df
    
    def summarize_parameters_with_llm(self, df):
        """
        使用LLM总结参数（已优化）
        """
        summarized = []

        # Warmup 自检：先尝试生成少量token，失败则立刻中止，避免写入空文件
        try:
            warmup_prompt = self._create_prompt(user_prompt="Reply with OK only.", context="warmup")
            warm = self.model.generate(prompt=warmup_prompt, max_tokens=8, temp=0.0)
            print(f"🔥 Warmup output: [{warm}] (len={len(warm) if warm else 0})")
            if not warm or not warm.strip():
                print("⚠️ Warmup 返回空，但继续尝试正常总结（可能模型需要更长prompt或特定参数）")
            else:
                print("🔥 Summarize warmup passed.")
        except Exception as e:
            print(f"⚠️ Warmup generate 异常: {e}，但继续尝试正常总结")
        
        # 1. 定义一个清晰的用户指令，包含所有规则和Schema
        # user_prompt = (
        #     "Extract structured data for flow-chemistry process development as a strict JSON object. "
        #     # 没有就返回null 避免幻觉
        #     "If a field is not explicitly stated, use null. Use original units when present; "
        #     # 在JSON里，像转化率、产率这些数值，请直接用数字格式
        #     "otherwise normalize as: temperature in °C, residence_time in min, flow_rate in mL/min, "
        #     "inner_diameter in mm. Use strings for values with units (e.g., \"100 °C\", \"0.20 mL/min\").\n\n"
        #     "### JSON Schema ###\n"
        #     "{\n"
        #     "  \"reaction_summary\": {\n"
        #     "    \"reaction_type\": \"...\", \n"
        #     "    \"reactants\": [ {\"name\": \"...\", \"role\": \"reactant|catalyst|solvent\"}, ... ],\n"
        #     "    \"products\": [ {\"name\": \"...\", \"yield_optimal\": 95, \"unit\": \"%\"}, ... ],\n"
        #     "    \"conditions\": [\n"
        #     "      {\"type\": \"temperature\", \"value\": \"...\"},\n"
        #     "      {\"type\": \"residence_time\", \"value\": \"...\"},\n"
        #     "      {\"type\": \"flow_rate_reactant_A\", \"value\": \"...\"},\n"
        #     "      {\"type\": \"flow_rate_total\", \"value\": \"...\"},\n"
        #     "      {\"type\": \"pressure\", \"value\": \"...\"}\n"
        #     "    ],\n"
        #     "    \"reactor\": {\"type\": \"...\", \"inner_diameter\": \"...\"},\n"
        #     "    \"metrics\": {\"conversion\": ..., \"yield\": ..., \"selectivity\": ..., \"unit\": \"%\"}\n"
        #     "  }\n"
        #     "}\n\n"
        #     "### Rules ###\n"
        #     # 只要纯净的json 不要任何多余文字
        #     "- Output ONLY the valid JSON object and nothing else (no introductory text or explanations).\n"
        #     "- Keep numbers as numbers where possible (e.g., in 'metrics'), but keep units within string values for 'conditions'.\n"
        #     # 只使用提供的段落作为证据，不要从其他部分推断，防止牛头马面 乱拼
        #     "- Only use the provided paragraph as evidence; do not infer from other parts of the paper.\n"
        #     # 只有最优选最优
        #     "- Set 'is_optimal': true only if words like 'optimal', 'optimized', 'best' are explicitly present in this paragraph; otherwise null.\n"
        #     # 没有最优选最高产率
        #     "- If multiple experimental conditions are reported, prioritize the one explicitly labeled as 'optimal'. If none are labeled, select the condition set that corresponds to the best reported performance (e.g., highest yield or conversion).\n"
        #     "- If multiple reactant streams have distinct flow rates, use specific keys like 'flow_rate_reactant_A', 'flow_rate_reactant_B', and include 'flow_rate_total' if it is also reported.\n"
        # ) # 过于严格了
        user_prompt = (
            "Only use the provided paragraph; do not infer across other paragraphs.\n"
            "If a field is not explicitly stated, use null. Use original units when present; "
            "otherwise normalize: temperature in °C, residence_time in min, flow_rate in mL/min, inner_diameter in mm.\n"
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
            "}}"
        )

        # 生成时降低随机性

        for index, row in df.iterrows():
            content = row['content']
            
            # 2. 使用辅助函数构建完整的Prompt
            full_prompt = self._create_prompt(user_prompt=user_prompt, context=content)
            
            try:
                # 3. 使用 self.model 进行调用
                # NEW: 更低随机性，便于严格JSON输出；首轮验证将 max_tokens 降至 300
                summarize_text = self.model.generate(prompt=full_prompt, max_tokens=300, temp=0.0, top_p=0.2)
                txt = (summarize_text or "").strip()
                # 轻后处理：若模型前后带说明文字，裁剪为最外层花括号包裹部分
                start, end = txt.find("{"), txt.rfind("}")
                if start != -1 and end != -1 and end > start:
                    txt = txt[start:end+1]
                print(f"Summarized {index+1}/{len(df)}:")        
                print(txt)
                summarized.append(txt)
            except Exception as e:
                print(f"Error generating summary for row {index}: {e}")
                summarized.append(f"Error: {e}")
        
        df['summarized'] = pd.Series(summarized, index=df.index) # 确保索引对齐
        return df

    def summarize_document_overall(self, df_abstract):
        """
        基于整篇抽象文本生成全局JSON总结（择优汇总，避免逐段丢项）
        输入: df_abstract，需包含列 'abstract' 或 'content'
        输出: 字符串JSON
        """
        import json

        col = 'abstract' if 'abstract' in df_abstract.columns else 'content'
        texts = [t for t in df_abstract[col].fillna("").tolist() if t.strip()]
        combined = "\n\n".join(texts)
        # 增加输入文本长度限制
        if len(combined) > 12000:
            combined = combined[:12000]

        # 极简 prompt，直接命令输出 JSON
        user_prompt = (
            "Extract flow chemistry parameters and output as ONE JSON object.\n\n"
            "CRITICAL RULES:\n"
            "- Output ONLY valid JSON - NO explanations, NO steps, NO markdown, NO comments\n"
            "- Do NOT write '### Step' or '```json' or any text outside JSON\n"
            "- Start directly with { and end with }\n"
            "- Use null for unknown fields (not \"...\" or \"not mentioned\")\n"
            "- Prefer best/optimal conditions when multiple values exist\n\n"
            "JSON format:\n"
            '{"reaction_summary": {\n'
            '  "reaction_type": "string or null",\n'
            '  "reactants": [{"name": "string", "role": "reactant|catalyst|solvent"}],\n'
            '  "products": [{"name": "string", "yield_optimal": number, "unit": "%"}],\n'
            '  "conditions": [\n'
            '    {"type": "temperature", "value": "string or null"},\n'
            '    {"type": "residence_time", "value": "string or null"},\n'
            '    {"type": "flow_rate_total", "value": "string or null"},\n'
            '    {"type": "pressure", "value": "string or null"}\n'
            '  ],\n'
            '  "reactor": {"type": "string or null", "inner_diameter": "string or null"},\n'
            '  "metrics": {"conversion": number, "yield": number, "selectivity": number, "unit": "%"}\n'
            "}}\n\n"
            "Output JSON now:"
        )

        full_prompt = self._create_prompt(user_prompt=user_prompt, context=combined)

        # 再做一次短warmup
        try:
            warm = self.model.generate(prompt="Say OK", max_tokens=8, temp=0.0)
            print(f"🔥 Overall warmup: [{warm}]")
        except Exception as e:
            print(f"⚠️ Overall warmup error: {e}（继续尝试）")

        # 增加 max_tokens 确保输出完整
        raw = self.model.generate(prompt=full_prompt, max_tokens=1200, temp=0.05, top_p=0.25) or ""
        raw = raw.strip()
        
        # 强力清洗：移除所有 markdown 代码块标记和步骤说明
        import re
        # 移除 ```json 和 ``` 标记
        raw = re.sub(r'```json\s*', '', raw)
        raw = re.sub(r'```\s*', '', raw)
        # 移除 ### Step 开头的行
        raw = re.sub(r'###\s+Step\s+\d+:.*?(?=\n|$)', '', raw, flags=re.MULTILINE)
        # 移除其他 markdown 标题
        raw = re.sub(r'###.*?(?=\n|$)', '', raw, flags=re.MULTILINE)

        # 提取最后一个完整的 JSON 对象（避免多个重复 JSON）
        # 策略：找到所有 {...} 块，取最大最完整的一个
        json_candidates = []
        brace_count = 0
        start_pos = -1
        for i, char in enumerate(raw):
            if char == '{':
                if brace_count == 0:
                    start_pos = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_pos != -1:
                    json_candidates.append(raw[start_pos:i+1])
                    start_pos = -1
        
        # 选择最长的候选（通常是最完整的）
        if json_candidates:
            raw = max(json_candidates, key=len)
        else:
            # 回退到原始逻辑
            s, e = raw.find("{"), raw.rfind("}")
            if s != -1 and e != -1 and e > s:
                raw = raw[s:e+1]
        # 清洗为合法JSON
        cleaned = self._sanitize_json_text(raw)

        # 校验
        try:
            json.loads(cleaned)
            return cleaned
        except Exception as _:
            print("⚠️ Overall JSON清洗后仍不合法，回退输出原文本以便排查。")
            return raw
            
    def _sanitize_json_text(self, text: str) -> str:
        """
        将可能含有注释/未加引号的键/尾随逗号的JSON样文本清洗为尽量合法的JSON字符串。
        """
        import re
        s = text or ""
        
        # 1. 移除所有非 JSON 的说明文字（如 "Note:", "Best regards" 等）
        # 找到第一个 { 和最后一个 }，只保留这之间的内容
        first_brace = s.find("{")
        last_brace = s.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            s = s[first_brace:last_brace+1]
        
        # 2. 去除 // 行内注释
        s = re.sub(r"//.*?(?=\n|$)", "", s)
        
        # 3. 去除 /* ... */ 块注释
        s = re.sub(r"/\*[\s\S]*?\*/", "", s)
        
        # 4. 修复未加引号的键
        s = re.sub(r'([\{,]\s*)([A-Za-z_][A-Za-z0-9_\-]*)\s*:\s*', r'\1"\2": ', s)
        
        # 5. 删除对象/数组中的尾随逗号
        s = re.sub(r",\s*(\}|\])", r"\1", s)
        
        # 6. 替换 ... 占位符为 null
        s = re.sub(r':\s*"\.\.\."\s*([,\}])', r': null\1', s)
        s = re.sub(r':\s*\.\.\.(\s*[,\}])', r': null\1', s)
        
        # 7. 修复可能的格式问题：确保数字不带引号
        s = re.sub(r':\s*"(\d+\.?\d*)"\s*([,\}])', r': \1\2', s)
        
        # 8. 去除多余的空白和换行
        s = s.strip()
        
        return s

    def extract_influence_factors_with_llm(self, df):
        """
        Extract influence factors and their impact on metrics with optimized batch processing
        
        Args:
            df: DataFrame with filtered/abstracted text content
            
        Returns:
            str: Markdown table format
        """
        print("\n🔬 Extracting influence factors (optimized, relaxed)...")
        
        # Phase 1: Smart candidate selection with scoring
        print("  📌 Phase 1: Intelligent candidate scoring...")
        candidates = self._extract_influence_candidates(df)
        
        if not candidates:
            print("  ⚠️ No candidate paragraphs found")
            return "| Factor | Metric | Direction | Magnitude/Unit | Condition | Evidence |\n|--------|--------|-----------|----------------|-----------|----------|\n| None detected | - | - | - | - | - |"
        
        print(f"  ✓ Selected {len(candidates)} top-scored candidates")
        
        # 🐛 DEBUG: 显示前3个候选段落
        print("\n  🐛 DEBUG - Top 3 candidates:")
        for i, cand in enumerate(candidates[:3]):
            print(f"    [{i}] Score={cand['score']}, Text preview: {cand['text'][:150]}...")
        print()
        
        # Phase 2: Batch LLM extraction (reduce API calls)
        print("  🤖 Phase 2: Batch LLM extraction...")
        all_items = self._batch_extract_with_llm(candidates)
        
        if not all_items:
            print("  ⚠️ LLM extraction produced no valid results")
            return "| Factor | Metric | Relationship |\n|--------|--------|--------------||\n| Extraction failed | - | - |"
        
        print(f"  ✓ Extracted {len(all_items)} raw items")
        
        # Phase 3: Normalize, deduplicate, filter noise
        print("  🔧 Phase 3: Normalizing and deduplicating...")
        normalized_items = self._normalize_impact_items(all_items)
        
        print(f"  ✓ {len(normalized_items)} items after normalization")

        # If still empty, try single-shot fallback (relaxed mode) with call limit guard
        if not normalized_items:
            import math
            group_size = int(os.getenv('FCPD_IMPACT_GROUP_SIZE', '5'))
            hard_limit = int(os.getenv('FCPD_IMPACT_HARD_LIMIT_CALLS', '3'))
            relaxed = os.getenv('FCPD_IMPACT_RELAXED', 'true').lower() == 'true'
            total_groups = math.ceil(len(candidates) / max(group_size, 1)) if candidates else 0
            if relaxed and total_groups < hard_limit:
                # Build a compact consolidated text from candidates (<= 1400 chars)
                joined = []
                used = 0
                for c in candidates:
                    t = c['text'].strip()
                    if not t:
                        continue
                    if len(t) > 240:
                        t = t[:240] + '...'
                    if used + len(t) + 2 > 1400:
                        break
                    joined.append(t)
                    used += len(t) + 2
                consolidated = "\n\n".join(joined)
                print("  🔁 Single-shot relaxed fallback (consolidated extraction)...")
                user_prompt = (
                    "From the following paragraphs, list cause-effect lines in the format: \n"
                    "Factor | Metric | Relationship\n\n"
                    "Guidelines:\n"
                    "- Use paper-specific factor names (e.g., 'Sulfuric acid strength', 'Residence time').\n"
                    "- If a metric is unchanged/no effect, write 'no effect' in Relationship.\n"
                    "- Prefer concise phrases. One line per relation.\n\n"
                    "Paragraphs:\n" + consolidated + "\n\n"
                    "Lines:"
                )
                full_prompt = self._create_prompt(user_prompt=user_prompt, context="")
                try:
                    raw = self.model.generate(prompt=full_prompt, max_tokens=400, temp=0.1) or ""
                    # Parse lines to items
                    items_fallback = self._parse_lines_to_items(raw)
                    normalized_items = self._normalize_impact_items(items_fallback)
                    print(f"  ✓ Fallback extracted {len(normalized_items)} items")
                except Exception as e:
                    print(f"  ⚠️ Fallback failed: {e}")
        
        # Phase 4: Format as Markdown table
        print("  📊 Phase 4: Formatting output...")
        markdown_table = self._to_markdown_impact(normalized_items)
        
        return markdown_table
    
    def _extract_influence_candidates(self, df):
        """
        Score and select top candidate paragraphs for impact extraction
        Uses adaptive scoring that prioritizes explicit causal patterns over generic keywords
        
        Returns:
            list of dict: [{'index': ..., 'text': ..., 'score': ...}, ...]
        """
        topK = int(os.getenv('FCPD_IMPACT_TOPK', '12'))
        content_col = 'abstract' if 'abstract' in df.columns else 'content'
        
        # Generic factor/metric keywords (lower weight - used as hints only)
        generic_factor_keywords = [
            'temperature', 'flow rate', 'residence time', 'pressure', 'concentration',
            'ratio', 'diameter', 'velocity', 'catalyst', 'solvent'
        ]
        
        metric_keywords = [
            'conversion', 'yield', 'selectivity', 'purity', 'efficiency',
            'product distribution', 'heat transfer', 'mixing'
        ]
        
        # Causal patterns (high weight - these indicate actual cause-effect discussion)
        causal_verbs = [
            'increase', 'decrease', 'improve', 'enhance', 'reduce', 'affect',
            'influence', 'impact', 'promote', 'inhibit', 'facilitate', 'optimize',
            'control', 'determine', 'depend', 'vary', 'change', 'modulate'
        ]
        
        # Explicit title patterns (highest weight)
        title_pattern = r'(\d+\.\d+\.?\d*\.?)\s*(Effect|Influence|Impact|Role)\s+of\s+([^.\n]{3,60})'
        
        # Causal relationship patterns (high weight - paper-specific factors)
        # These capture: "X affects/influences Y" or "Y depends on X"
        causal_patterns = [
            r'([A-Z][a-z\s]{2,40})\s+(affect|influence|impact|control|determine)s?\s+(?:the\s+)?([a-z\s]{3,30})',
            r'([a-z\s]{3,30})\s+(?:is|was|are|were)\s+(?:strongly\s+)?(?:affected|influenced|controlled|determined)\s+by\s+([A-Z][a-z\s]{2,40})',
            r'(?:higher|lower|increased|decreased)\s+([a-z\s]{2,30})\s+(?:result|lead|cause)s?\s+(?:in\s+)?(?:higher|lower|increased|decreased)\s+([a-z\s]{2,30})',
            r'at\s+([0-9.]+\s*[A-Za-z°%]+)[,\s]+(?:the\s+)?([a-z\s]{3,30})\s+(?:is|was|reached)'
        ]
        
        candidates = []
        
        for idx, row in df.iterrows():
            paragraph = row[content_col]
            if not paragraph or len(paragraph.strip()) < 30:
                continue
            
            score = 0
            hits = []
            para_lower = paragraph.lower()
            
            # 1. HIGHEST PRIORITY: Explicit "Effect of X" titles
            title_match = re.search(title_pattern, paragraph, re.IGNORECASE)
            if title_match:
                score += 80  # Increased from 50
                factor_name = title_match.group(3).strip()
                hits.append(f'title:{factor_name[:20]}')
            
            # 2. HIGH PRIORITY: Explicit causal relationship patterns
            # Only apply if paragraph has causal verbs (pre-filter for performance)
            has_causal_verb = any(verb in para_lower for verb in ['affect', 'influence', 'impact', 'control', 'determine'])
            if has_causal_verb:
                # Limit to 2 most important patterns for performance
                key_patterns = causal_patterns[:2]  # Only first 2 patterns
                for pattern in key_patterns:
                    if re.search(pattern, paragraph, re.IGNORECASE):
                        score += 15
                        hits.append('causal_pattern')
                        break  # Stop after first match
            
            # 3. MEDIUM PRIORITY: Results/Discussion sections with metrics
            # Simple string matching (faster than regex)
            is_results_section = any(sec in para_lower for sec in ['result', 'discussion', 'finding', 'observation'])
            if is_results_section:
                has_metric = any(kw in para_lower for kw in metric_keywords)
                if has_metric:
                    score += 25
                    hits.append('results_with_metrics')
                else:
                    score += 10
                    hits.append('results_section')
            
            # 4. MEDIUM PRIORITY: Causal verbs with metrics nearby
            # Simplified: just check if causal verb AND metric both present
            causal_count = sum(1 for verb in causal_verbs if verb in para_lower)
            if causal_count > 0:
                has_metric = any(m in para_lower for m in metric_keywords)
                if has_metric:
                    score += 8 * min(causal_count, 3)  # Cap at 3 to avoid over-scoring
                else:
                    score += 3 * min(causal_count, 2)  # Cap at 2
                hits.append(f'{causal_count}_causal_verbs')
            
            # 5. LOW PRIORITY: Generic keywords (used as tie-breakers only)
            # Only count if there's already some causal signal
            if score > 0:
                generic_factor_count = sum(1 for kw in generic_factor_keywords if kw in para_lower)
                metric_count = sum(1 for kw in metric_keywords if kw in para_lower)
                
                score += generic_factor_count * 2  # Low weight
                score += metric_count * 3  # Slightly higher for metrics
                
                if generic_factor_count > 0:
                    hits.append(f'{generic_factor_count}_generic_factors')
                if metric_count > 0:
                    hits.append(f'{metric_count}_metrics')
            
            # 6. BOOST: Quantitative data present (numbers with units)
            # This helps identify paragraphs with actual experimental conditions
            quant_pattern = r'\b\d+\.?\d*\s*(?:%|K|°C|°F|mL|L|min|h|MPa|bar|M|mol|g)\b'
            quant_matches = re.findall(quant_pattern, paragraph)
            if len(quant_matches) >= 2:
                score += 10
                hits.append(f'{len(quant_matches)}_quantitative')
            
            # 7. PENALTY: Too generic or irrelevant
            # Reduce score if paragraph is too short or has no specific content
            if len(paragraph) < 100:
                score -= 5
            
            # Must have at least some relevance (lowered threshold for adaptive discovery)
            if score >= 8:  # Lowered from 10
                candidates.append({
                    'index': idx,
                    'text': paragraph,
                    'score': score,
                    'hits': hits
                })
        
        # Sort by score and take topK
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates[:topK]
    
    def _batch_extract_with_llm(self, candidates):
        """
        Process candidates in batches to reduce LLM calls
        
        Returns:
            list of dict: extracted items with structured fields
        """
        group_size = int(os.getenv('FCPD_IMPACT_GROUP_SIZE', '5'))
        max_tokens_per_group = int(os.getenv('FCPD_IMPACT_MAX_TOKENS_PER_GROUP', '700'))
        relaxed = os.getenv('FCPD_IMPACT_RELAXED', 'true').lower() == 'true'
        
        all_items = []
        total_groups = (len(candidates) + group_size - 1) // group_size
        
        for g_idx in range(total_groups):
            group = candidates[g_idx * group_size : (g_idx + 1) * group_size]
            
            # Build batch prompt
            para_list = []
            for i, cand in enumerate(group):
                text = cand['text']
                if len(text) > 600:
                    text = text[:600] + "..."
                para_list.append(f"Para[{i}]: {text}")
            
            # Few-shot example with more realistic text
            fewshot_text = (
                "2.1.2. Eﬀect of Residence Time. Residence time is another important parameter which aﬀects the reagent conversion and "
                "product selectivity. [...] It is observed that a longer residence time will result in a higher conversion of "
                "TFMB, especially with mixed acid of a higher sulfuric acid strength. Nevertheless, the product selectivity "
                "nearly remains unchanged with the increasing residence time during the experiments..."
            )
            fewshot_json = """[
  {
    "i": 0,
    "factor": "residence time",
    "metric": "conversion of TFMB",
    "direction": "increase",
    "magnitude": null,
    "unit": null,
    "condition": "especially with mixed acid of a higher sulfuric acid strength",
    "evidence": "a longer residence time will result in a higher conversion of TFMB"
  },
  {
    "i": 0,
    "factor": "residence time",
    "metric": "product selectivity",
    "direction": null,
    "magnitude": null,
    "unit": null,
    "condition": null,
    "evidence": "the product selectivity nearly remains unchanged"
  }
]"""
            
            # Build structured prompt with clear schema and rules
            user_prompt = f"""Extract ALL cause-and-effect relationships from the provided paragraphs about chemical engineering processes.

### JSON Schema ###
- "i": The 0-based index of the paragraph the information was extracted from.
- "factor": The exact name of the process parameter that is being changed (e.g., "residence time", "reaction temperature").
- "metric": The experimental outcome that is being affected (e.g., "conversion of TFMB", "product selectivity").
- "direction": The direction of the effect. Must be one of: "increase", "decrease", or null if there is no effect or the direction is unclear.
- "magnitude": Any specific number mentioned, otherwise null.
- "unit": The unit for the magnitude, otherwise null.
- "condition": Any specific condition under which this effect was observed, otherwise null (e.g., "at high temperatures").
- "evidence": A short, direct quote from the text that supports the finding (max 100 characters).

### High-Quality Example ###
Context: "{fewshot_text}"
JSON Output:
{fewshot_json}

### Rules ###
- Output ONLY a valid JSON array. Do not include any explanations or surrounding text.
- Extract relationships only from the provided paragraphs.
- If a paragraph contains multiple relationships, create a separate JSON object for each.
- If a paragraph contains no relationships, output an empty array [].

### Paragraphs to Analyze ###
{chr(10).join(para_list)}

### Final JSON Output ###
"""
            
            full_prompt = self._create_prompt(user_prompt=user_prompt, context="")
            
            try:
                raw = self.model.generate(prompt=full_prompt, max_tokens=max_tokens_per_group, temp=0.05, top_p=0.25) or ""
                raw = raw.strip()
                
                # 🐛 DEBUG: 打印LLM原始输出
                print(f"\n    🐛 DEBUG Group {g_idx+1} - LLM raw output (first 500 chars):")
                print(f"    {raw[:500]}")
                print(f"    🐛 DEBUG - Total length: {len(raw)} chars\n")
                
                # Extract JSON array
                start = raw.find('[')
                end = raw.rfind(']')
                if start != -1 and end != -1 and end > start:
                    json_str = raw[start:end+1]
                    json_str = self._sanitize_json_text(json_str)
                    
                    import json
                    items = json.loads(json_str)
                    if isinstance(items, list):
                        all_items.extend(items)
                        print(f"    ✓ Group {g_idx+1}/{total_groups}: extracted {len(items)} items")
                    else:
                        print(f"    ⚠️ Group {g_idx+1}/{total_groups}: invalid JSON structure")
                else:
                    # Relaxed parsing: try line-based fallback
                    if relaxed:
                        items_lines = self._parse_lines_to_items(raw)
                        if items_lines:
                            all_items.extend(items_lines)
                            print(f"    ✓ Group {g_idx+1}/{total_groups}: line-based extracted {len(items_lines)} items")
                        else:
                            # Regex fallback per paragraph (no extra LLM calls)
                            regex_items = []
                            for c in group:
                                regex_items.extend(self._regex_extract_items_from_paragraph(c['text']))
                            if regex_items:
                                all_items.extend(regex_items)
                                print(f"    ✓ Group {g_idx+1}/{total_groups}: regex fallback {len(regex_items)} items")
                            else:
                                print(f"    ⚠️ Group {g_idx+1}/{total_groups}: no JSON/lines/regex matches")
                    else:
                        print(f"    ⚠️ Group {g_idx+1}/{total_groups}: no JSON array found")
            
            except Exception as e:
                print(f"    ✗ Group {g_idx+1}/{total_groups} failed: {e}")
                continue
        
        return all_items
    
    def _normalize_impact_items(self, items):
        """
        Normalize terminology lightly, deduplicate, and filter noise
        Preserves paper-specific factor names (e.g., "Sulfuric acid strength", "Sand particle size")
        
        Returns:
            list of dict: cleaned and normalized items
        """
        # Light normalization - only fix common typos/abbreviations
        # DO NOT replace paper-specific terms
        factor_abbreviations = {
            'temp': 'temperature',
            'conc': 'concentration',
            'res time': 'residence time'
        }
        
        metric_standardization = {
            'conversion': 'Conversion',
            'yield': 'Yield',
            'selectivity': 'Selectivity',
            'purity': 'Purity',
            'efficiency': 'Efficiency',
            'product distribution': 'Product Distribution',
            'heat transfer': 'Heat Transfer',
            'mixing': 'Mixing Efficiency'
        }
        
        normalized = []
        seen_pairs = set()
        
        for item in items:
            if not isinstance(item, dict):
                continue
            
            factor = str(item.get('factor', '')).strip()
            metric = str(item.get('metric', '')).strip()
            
            # Filter noise
            if not factor or not metric:
                continue
            if factor.lower() in ['multiple factors', 'unknown', 'various', 'x', 'parameter']:
                continue
            if metric.lower() in ['performance metrics', 'unknown', 'various', 'y', 'result']:
                continue
            if len(factor) < 3 or len(metric) < 3:
                continue
            
            # Light normalization for factors (preserve paper-specific names)
            factor_lower = factor.lower()
            for abbrev, full in factor_abbreviations.items():
                if factor_lower == abbrev:  # Exact match only
                    factor = full.title()
                    break
            
            # Always capitalize first letter of factor for consistency
            if factor and not factor[0].isupper():
                factor = factor[0].upper() + factor[1:]
            
            # Standardize metrics (these are more generic)
            metric_lower = metric.lower()
            for key, val in metric_standardization.items():
                if key == metric_lower:  # Exact match
                    metric = val
                    break
            
            # Capitalize metric if not already standardized
            if metric and not metric[0].isupper():
                metric = metric[0].upper() + metric[1:]
            
            # Deduplicate by factor-metric pair
            pair_key = (factor.lower(), metric.lower())
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            
            # Build normalized item
            direction = str(item.get('direction', '')).lower()
            if direction not in ['increase', 'decrease']:
                direction = ''
            
            magnitude = item.get('magnitude')
            unit = str(item.get('unit', '')).strip()
            condition = str(item.get('condition', '')).strip()
            evidence = str(item.get('evidence', '')).strip()
            
            normalized.append({
                'factor': factor,
                'metric': metric,
                'direction': direction,
                'magnitude': magnitude,
                'unit': unit,
                'condition': condition,
                'evidence': evidence[:100] if evidence else ''  # Truncate
            })
        
        return normalized
    
    def _to_markdown_impact(self, items):
        """
        Convert structured items to Markdown table
        
        Returns:
            str: Markdown table
        """
        if not items:
            return "| Factor | Metric | Direction | Magnitude/Unit | Condition | Evidence |\n|--------|--------|-----------|----------------|-----------|----------|\n| None | - | - | - | - | - |"
        
        # Build table with extended columns
        table_lines = [
            "| Factor | Metric | Direction | Magnitude/Unit | Condition | Evidence |",
            "|--------|--------|-----------|----------------|-----------|----------|"
        ]
        
        for item in items:
            factor = item['factor'].replace('|', '\\|')
            metric = item['metric'].replace('|', '\\|')
            direction = ('↑' if item['direction'] == 'increase' else 
                        '↓' if item['direction'] == 'decrease' else '-')
            
            mag_unit = ''
            if item['magnitude'] is not None:
                mag_unit = f"{item['magnitude']}"
                if item['unit']:
                    mag_unit += f" {item['unit']}"
            else:
                mag_unit = '-'
            
            condition = item['condition'] if item['condition'] else '-'
            condition = condition.replace('|', '\\|')[:40]  # Truncate
            
            evidence = item['evidence'] if item['evidence'] else '-'
            evidence = evidence.replace('|', '\\|').replace('\n', ' ')[:60]  # Truncate
            
            table_lines.append(
                f"| {factor} | {metric} | {direction} | {mag_unit} | {condition} | {evidence} |"
            )
        
        return '\n'.join(table_lines)

    def _parse_lines_to_items(self, text):
        """
        Parse loose 'Factor | Metric | Relationship' lines to structured items
        """
        items = []
        for line in (text or '').split('\n'):
            if '|' not in line:
                continue
            parts = [p.strip() for p in line.split('|')]
            if len(parts) < 3:
                continue
            factor, metric, relation = parts[0], parts[1], parts[2]
            if not factor or not metric:
                continue
            # Detect direction from relation text
            rel_low = relation.lower()
            if any(k in rel_low for k in ['no effect', 'unchanged', 'not affected', 'no significant']):
                direction = ''
            elif any(k in rel_low for k in ['increase', 'higher', 'improve', 'enhance']):
                direction = 'increase'
            elif any(k in rel_low for k in ['decrease', 'lower', 'reduce', 'inhibit']):
                direction = 'decrease'
            else:
                direction = ''
            items.append({
                'factor': factor,
                'metric': metric,
                'direction': direction,
                'magnitude': None,
                'unit': '',
                'condition': '',
                'evidence': relation[:100]
            })
        return items

    def _regex_extract_items_from_paragraph(self, paragraph):
        """
        Zero-LLM fallback: use regex to extract simple X→Y relations
        """
        results = []
        p = paragraph or ''
        if len(p) < 30:
            return results
        import re
        # Pattern: higher X increases Y / lower X decreases Y
        pat1 = re.compile(r'(higher|lower|increased|decreased)\s+([A-Za-z][A-Za-z\s]{2,30})\s+(?:lead|leads|result|results|cause|causes)\s+(?:in\s+)?(higher|lower|increased|decreased)\s+([A-Za-z][A-Za-z\s]{2,30})', re.IGNORECASE)
        m = pat1.search(p)
        if m:
            dir_map = {'higher':'increase','increased':'increase','lower':'decrease','decreased':'decrease'}
            factor = m.group(2).strip()
            metric = m.group(4).strip()
            direction = dir_map.get(m.group(3).lower(), '')
            results.append({'factor':factor, 'metric':metric, 'direction':direction, 'magnitude':None, 'unit':'', 'condition':'', 'evidence': p[:100]})
        # Pattern: X affects Y / Y affected by X
        pat2 = re.compile(r'([A-Za-z][A-Za-z\s]{2,40})\s+(affect|influence|impact|control|determine)s?\s+([A-Za-z][A-Za-z\s]{2,40})', re.IGNORECASE)
        m2 = pat2.search(p)
        if m2:
            results.append({'factor':m2.group(1).strip(), 'metric':m2.group(3).strip(), 'direction':'', 'magnitude':None, 'unit':'', 'condition':'', 'evidence': p[:100]})
        # No effect phrases
        if re.search(r'(no effect|unchanged|not significantly affected)', p, re.IGNORECASE):
            # Try to attach to the last added metric if present
            if results:
                results[-1]['direction'] = ''
        return results
    
    def save_df_to_text(self, df, file_path, content_column='content'):
        """
        保存DataFrame到文本文件
        """
        with open(file_path, 'w', encoding='utf-8') as file:
            for index, row in df.iterrows():
                file.write(row[content_column] + '\n\n')
    
    def process_text_file_comprehensive(self, file_path, mode='comprehensive'):
        """
        综合文本处理函数
        mode: 'filter' - 只过滤
              'abstract' - 只抽象
              'summarize' - 只总结
              'comprehensive' - 完整流程
        """
        print(f"🔍 处理文件: {os.path.basename(file_path)}")
        print("=" * 50)
        
        # 读取文件
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        # 分割段落
        current_segment = []
        segments = []
        
        for line in lines:
            if line.strip():
                current_segment.append(line.strip())
            else:
                if current_segment:
                    segments.append(' '.join(current_segment))
                    current_segment = []
        
        if current_segment:
            segments.append(' '.join(current_segment))
        
        df = pd.DataFrame(segments, columns=['content'])
        print(f"📊 原始段落数: {len(df)}")
        
        output_files = {}
        
        if mode in ['filter', 'comprehensive']:
            # 1. LLM内容过滤
            print("\n🤖 步骤1: LLM内容过滤...")
            df_filtered = self.filter_content_with_llm(df)
            print(f"✅ 过滤后段落数: {len(df_filtered)}")
            
            # 保存过滤结果
            filter_file = file_path.replace('.txt', '_Filtered.txt')
            self.save_df_to_text(df_filtered, filter_file)
            output_files['filter'] = filter_file
        
        if mode in ['abstract', 'comprehensive']:
            # 2. 文本抽象
            print("\n📝 步骤2: 文本抽象...")
            df_abstract = self.abstract_text_with_llm(df_filtered if 'df_filtered' in locals() else df)
            
            # 保存抽象结果
            abstract_file = file_path.replace('.txt', '_Abstract.txt')
            self.save_df_to_text(df_abstract, abstract_file, 'abstract')
            output_files['abstract'] = abstract_file
        
        if mode in ['summarize', 'comprehensive']:
            # 3. 参数总结
            print("\n📊 步骤3: 参数总结...")
            # OLD: 直接使用过滤后的原始段落进行总结
            # df_summarized = self.summarize_parameters_with_llm(df_filtered if 'df_filtered' in locals() else df)

            # NEW: 优先使用抽象后的文本作为总结输入；若无抽象则退回过滤文本，再退回原始文本
            # 补充：当 mode='summarize' 且本次未运行抽象步骤时，尝试从同名抽象文件加载抽象结果
            if 'df_abstract' not in locals():
                try:
                    abstract_file_try = file_path.replace('.txt', '_Abstract.txt')
                    if os.path.exists(abstract_file_try):
                        with open(abstract_file_try, 'r', encoding='utf-8', errors='ignore') as f_abs:
                            lines_abs = f_abs.readlines()
                        current_segment_abs = []
                        segments_abs = []
                        for line in lines_abs:
                            if line.strip():
                                current_segment_abs.append(line.strip())
                            else:
                                if current_segment_abs:
                                    segments_abs.append(' '.join(current_segment_abs))
                                    current_segment_abs = []
                        if current_segment_abs:
                            segments_abs.append(' '.join(current_segment_abs))
                        if segments_abs:
                            df_abstract = pd.DataFrame(segments_abs, columns=['content'])
                            print(f"🔁 载入已有抽象文件用于总结: {os.path.basename(abstract_file_try)}，段落数: {len(df_abstract)}")
                except Exception as e:
                    print(f"⚠️ 载入抽象文件失败，改用过滤或原始文本: {e}")

            df_input_for_sum = df_abstract if 'df_abstract' in locals() else (df_filtered if 'df_filtered' in locals() else df)
            df_summarized = self.summarize_parameters_with_llm(df_input_for_sum)
            
            # 保存总结结果
            summarize_file = file_path.replace('.txt', '_Summarized.txt')
            self.save_df_to_text(df_summarized, summarize_file, 'summarized')
            output_files['summarized'] = summarize_file

            # 整篇汇总总结（以抽象为输入优先）
            try:
                overall_input = df_abstract if 'df_abstract' in locals() else (df_filtered if 'df_filtered' in locals() else df)
                overall_json = self.summarize_document_overall(overall_input)
                overall_file = file_path.replace('.txt', '_Overall.txt')
                with open(overall_file, 'w', encoding='utf-8') as f:
                    f.write(overall_json)
                output_files['summarized_overall'] = overall_file
                print(f"  🧩 整篇汇总总结完成: {os.path.basename(overall_file)}")
            except Exception as e:
                print(f"⚠️ 整篇汇总总结失败: {e}")
            
            # 提取影响因素分析（新功能）
            try:
                influence_input = df_abstract if 'df_abstract' in locals() else (df_filtered if 'df_filtered' in locals() else df)
                
                # 调试：保存Impact输入数据
                debug_input_file = file_path.replace('.txt', '_Impact_Input_Debug.txt')
                self.save_df_to_text(influence_input, debug_input_file, 'content')
                print(f"  🐛 DEBUG: 影响因素输入已保存到 {os.path.basename(debug_input_file)}")
                
                influence_md = self.extract_influence_factors_with_llm(influence_input)
                influence_file = file_path.replace('.txt', '_Impact_Analysis.txt')
                with open(influence_file, 'w', encoding='utf-8') as f:
                    f.write("# Influence Factor Analysis\n\n")
                    f.write(influence_md)
                output_files['impact_analysis'] = influence_file
                print(f"  📊 影响因素分析完成: {os.path.basename(influence_file)}")
            except Exception as e:
                print(f"⚠️ 影响因素分析失败: {e}")
            
            # 清理中间文件，只保留最终的 _Summarized.txt、_Overall.txt 和 _Impact_Analysis.txt
            # 🐛 DEBUG: 临时禁用自动清理，便于调试
            auto_cleanup_enabled = os.getenv('FCPD_AUTO_CLEANUP', 'false').lower() == 'true'
            
            if not auto_cleanup_enabled:
                print("\n🐛 DEBUG: 自动清理已禁用，保留所有中间文件")
                return output_files
            
            print("\n🗑️  清理中间文件...")
            intermediate_files = []
            
            # 获取文件所在目录和基础文件名
            file_dir = os.path.dirname(file_path)
            # 从文件名中提取原始PDF基础名（移除 Embedding_ 前缀和所有后缀）
            file_name = os.path.basename(file_path)
            if file_name.startswith('Embedding_'):
                file_base = file_name.replace('Embedding_', '').split('_Filtered')[0].split('_Abstract')[0].split('.txt')[0]
            else:
                file_base = file_name.replace('.txt', '')
            
            # 需要删除的所有中间文件（按处理流程顺序）
            # 1. 原始PDF转文本
            intermediate_files.append(os.path.join(file_dir, f"{file_base}.txt"))
            intermediate_files.append(os.path.join(file_dir, f"{file_base}_other.txt"))
            
            # 2. 文本预处理
            intermediate_files.append(os.path.join(file_dir, f"Processed_{file_base}.txt"))
            
            # 3. 嵌入相似度筛选
            intermediate_files.append(os.path.join(file_dir, f"Embedding_{file_base}.txt"))
            
            # 4. LLM内容过滤
            intermediate_files.append(os.path.join(file_dir, f"Embedding_{file_base}_Filtered.txt"))
            
            # 5. 文本抽象
            intermediate_files.append(os.path.join(file_dir, f"Embedding_{file_base}_Filtered_Abstract.txt"))
            
            # 6. 总结表格文件
            intermediate_files.append(os.path.join(file_dir, f"Embedding_{file_base}_Filtered_Abstract_Summarized.tsv"))
            intermediate_files.append(os.path.join(file_dir, f"Embedding_{file_base}_Filtered_Abstract_Summarized.md"))
            
            # 删除中间文件
            deleted_count = 0
            for f in intermediate_files:
                if os.path.exists(f):
                    try:
                        os.remove(f)
                        print(f"    ✓ 已删除: {os.path.basename(f)}")
                        deleted_count += 1
                    except Exception as e:
                        print(f"    ✗ 删除失败 {os.path.basename(f)}: {e}")
            
            if deleted_count > 0:
                print(f"  🎉 清理完成，删除了 {deleted_count} 个中间文件")
            
            # 显示保留的最终文件
            print(f"\n📦 最终保留文件:")
            if 'summarized' in output_files:
                print(f"  • {os.path.basename(output_files['summarized'])} (详细总结)")
            if 'summarized_overall' in output_files:
                print(f"  • {os.path.basename(output_files['summarized_overall'])} (整篇汇总)")
            if 'impact_analysis' in output_files:
                print(f"  • {os.path.basename(output_files['impact_analysis'])} (影响因素分析)")
        
        return output_files

# 兼容性函数
def process_text_file_for_filter(file_path, model_name='nous-hermes-llama2-13b.Q4_0.gguf'):
    """
    LLM内容过滤函数
    Args:
        file_path: 要处理的文件路径
        model_name: 使用的LLM模型名称
    """
    processor = UnifiedTextProcessor(model_name=model_name)
    result = processor.process_text_file_comprehensive(file_path, mode='filter')
    return list(result.values())[0]

def process_text_file_for_abstract(file_path, model_name='nous-hermes-llama2-13b.Q4_0.gguf'):
    """
    文本抽象函数
    Args:
        file_path: 要处理的文件路径
        model_name: 使用的LLM模型名称
    """
    processor = UnifiedTextProcessor(model_name=model_name)
    result = processor.process_text_file_comprehensive(file_path, mode='abstract')
    return list(result.values())[0]

def process_text_file_for_summerized(file_path, model_name='meta-llama-3.1-8b-instruct-q4_k_m-2.gguf', strict=True):
    """
    参数总结函数
    Args:
        file_path: 要处理的文件路径
        model_name: 使用的LLM模型名称
        strict: 是否严格使用指定模型（不回退）
    """
    processor = UnifiedTextProcessor(model_name=model_name, strict=strict)
    result = processor.process_text_file_comprehensive(file_path, mode='summarize')
    return list(result.values())[0]

# 使用Meta-Llama模型的专用函数
def process_text_file_for_filter_meta_llama(file_path):
    processor = UnifiedTextProcessor(model_name='Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf')
    result = processor.process_text_file_comprehensive(file_path, mode='filter')
    return list(result.values())[0]

def process_text_file_for_abstract_meta_llama(file_path):
    processor = UnifiedTextProcessor(model_name='Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf')
    result = processor.process_text_file_comprehensive(file_path, mode='abstract')
    return list(result.values())[0]

def process_text_file_for_summerized_meta_llama(file_path):
    processor = UnifiedTextProcessor(model_name='Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf')
    result = processor.process_text_file_comprehensive(file_path, mode='summarize')
    return list(result.values())[0]

# 严格：仅用 Meta-Llama 做 summarize，模型加载失败不回退
def process_text_file_for_summerized_meta_llama_strict(file_path):
    processor = UnifiedTextProcessor(model_name='Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf', strict=True)
    result = processor.process_text_file_comprehensive(file_path, mode='summarize')
    return list(result.values())[0]
