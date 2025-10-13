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
    
    def __init__(self, model_name='nous-hermes-llama2-13b.Q4_0.gguf', model_path='models/'):
        self.model_name = model_name
        self.model_path = model_path
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        
    def load_llm_model(self):
        """
        加载LLM模型，支持回退机制
        """
        # 获取绝对路径
        abs_model_path = os.path.abspath(self.model_path)
        print(f"🔍 尝试加载模型，路径: {abs_model_path}")
        
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
    
    def filter_content_with_llm(self, df):
        """
        使用LLM过滤内容（替代Filter.py功能）
        """
        model = self.load_llm_model()
        
        questions = [
            "Question: Does this section cover the types of surface chemical reactions or experimental studies on the formation of molecules on surfaces? Answer 'Yes' or 'No'. \nAnswer:"
        ]
        
        for idx, row in df.iterrows():
            content = row['content']
            classification = 'No'
            
            for question in questions:
                prompt = f"{content}\n{question}"
                try:
                    response = model.generate(prompt=prompt, max_tokens=10, temp=0.1)
                    if response and response.strip():
                        first_word = response.split()[0].replace('.', '').replace(',', '')
                        if first_word not in ['No', 'Not']:
                            classification = first_word
                            break
                except Exception as e:
                    print(f"Error generating response: {e}")
                    continue
            
            df.loc[idx, 'classification'] = classification
        
        # 过滤掉"No"的段落
        condition = (df['classification'] != 'No') & (df['classification'] != 'Not')
        df_filtered = df[condition]
        
        return df_filtered
    
    def create_abstract_conclusion_embeddings(self, df):
        """
        创建摘要和结论专用的嵌入（整合Abstract_Conclusion_Embedding.py功能）
        """
        # 定义摘要和结论相关的关键词
        abstract_conclusion_keywords = [
            "conclusion", "abstract", "summary", "findings", "results", 
            "synthesis parameters", "reaction conditions", "precursor molecules",
            "substrate", "temperature", "products", "experimental results",
            "key findings", "main results", "final results", "outcome",
            "synthesis", "reaction", "molecular", "surface chemistry",
            "on-surface", "catalytic", "formation", "yield", "selectivity"
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
        使用LLM进行文本抽象（整合Abstract.py功能）
        """
        model = self.load_llm_model()
        abstract = []
        
        for index, row in df.iterrows():
            content = row['content']
            
            prompt_template = (
                f"{content}"
                f"Answer the question as truthfully as possible using the provided context."
                f"Please summarize the text below, emphasizing the types of reactions featured in the author's scientific experiments on surface reactions."            
            )
            
            try:
                abstract_text = model.generate(prompt=prompt_template, max_tokens=250, temp=0.0, top_p=0.6)
                print(f"Abstract {index+1}/{len(df)}:")
                print(abstract_text)
                abstract.append(abstract_text)
            except Exception as e:
                print(f"Error generating abstract: {e}")
                abstract.append("Error generating abstract")
        
        df['abstract'] = pd.Series(abstract)
        return df
    
    def summarize_parameters_with_llm(self, df):
        """
        使用LLM总结参数（整合Summerized.py功能）
        """
        model = self.load_llm_model()
        summarized = []
        
        for index, row in df.iterrows():
            content = row['content']
            prompt_template = (
                f"{content}\n"            
                f"Task: Please summarize the following details in a table: precursor molecules, substrates, annealing/reaction temperature of the molecules, products (i.e., the compound molecules formed in this experiment), and the dimensionality of the product molecules (Simplified numbers plus letters). If no information is provided or you are unsure, use N/A. Please focus on extracting experimental conditions only from the surface chemistry synthesis. The table should have 5 columns: | Precursor | Substrate | Temperature | Products | Dimensions |"
            )
            
            try:
                summarize_text = model.generate(prompt=prompt_template, max_tokens=250, temp=0.0, top_p=0.6)
                print(f"Summarized {index+1}/{len(df)}:")        
                print(summarize_text)
                summarized.append(summarize_text)
            except Exception as e:
                print(f"Error generating summary: {e}")
                summarized.append("Error generating summary")
        
        df['summarized'] = pd.Series(summarized)
        return df
    
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
            df_summarized = self.summarize_parameters_with_llm(df_filtered if 'df_filtered' in locals() else df)
            
            # 保存总结结果
            summarize_file = file_path.replace('.txt', '_Summarized.txt')
            self.save_df_to_text(df_summarized, summarize_file, 'summarized')
            output_files['summarized'] = summarize_file
        
        return output_files

# 兼容性函数
def process_text_file_for_filter(file_path):
    processor = UnifiedTextProcessor()
    result = processor.process_text_file_comprehensive(file_path, mode='filter')
    return list(result.values())[0]

def process_text_file_for_abstract(file_path):
    processor = UnifiedTextProcessor()
    result = processor.process_text_file_comprehensive(file_path, mode='abstract')
    return list(result.values())[0]

def process_text_file_for_summerized(file_path):
    processor = UnifiedTextProcessor()
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
