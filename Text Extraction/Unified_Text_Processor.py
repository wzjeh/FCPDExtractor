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
        if len(combined) > 8000:
            combined = combined[:8000]

        user_prompt = (
            "You will be given multiple paragraph-level abstracts of a single paper (flow chemistry).\n"
            "Aggregate them into ONE consolidated JSON object that captures the overall best/representative conditions.\n"
            "- Use only info present in the abstracts (do not invent).\n"
            "- If multiple values exist, prefer ones explicitly marked as optimal/best; otherwise pick those associated with highest performance (yield/conversion/selectivity).\n"
            "- Keep units as-is when present.\n"
            "Output ONLY the JSON (no extra text):\n"
            "{ \"reaction_summary\": {\n"
            "  \"reaction_type\": \"...\",\n"
            "  \"reactants\": [ {\"name\": \"...\", \"role\": \"reactant|catalyst|solvent\"} ],\n"
            "  \"products\": [ {\"name\": \"...\", \"yield_optimal\": 95, \"unit\": \"%\"} ],\n"
            "  \"conditions\": [\n"
            "     {\"type\": \"temperature\", \"value\": \"...\"},\n"
            "     {\"type\": \"residence_time\", \"value\": \"...\"},\n"
            "     {\"type\": \"flow_rate_total\", \"value\": \"...\"},\n"
            "     {\"type\": \"pressure\", \"value\": \"...\"}\n"
            "  ],\n"
            "  \"reactor\": {\"type\": \"...\", \"inner_diameter\": \"...\"},\n"
            "  \"metrics\": {\"conversion\": ..., \"yield\": ..., \"selectivity\": ..., \"unit\": \"%\"}\n"
            "} }\n"
        )

        full_prompt = self._create_prompt(user_prompt=user_prompt, context=combined)

        # 再做一次短warmup
        try:
            warm = self.model.generate(prompt="Say OK", max_tokens=8, temp=0.0)
            print(f"🔥 Overall warmup: [{warm}]")
        except Exception as e:
            print(f"⚠️ Overall warmup error: {e}（继续尝试）")

        raw = self.model.generate(prompt=full_prompt, max_tokens=500, temp=0.0, top_p=0.2) or ""
        raw = raw.strip()

        # 裁剪到最外层花括号
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
        # 去除 // 行内注释
        s = re.sub(r"//.*?(?=\n|$)", "", s)
        # 去除 /* ... */ 注释
        s = re.sub(r"/\*[\s\S]*?\*/", "", s)
        # 修复未加引号的键: 在 { 或 , 之后出现的裸键名
        s = re.sub(r"([\{,]\s*)([A-Za-z_][A-Za-z0-9_\-]*)\s*:\s*", r'\1"\2": ', s)
        # 删除对象/数组中的尾随逗号
        s = re.sub(r",\s*(\}|\])", r"\1", s)
        # 去除多余空白
        s = s.strip()
        return s

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
            
            # 清理中间文件，只保留最终的 _Summarized.txt 和 _Overall.txt
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
