import fitz  # PyMuPDF
import pandas as pd
import re
import os
import spacy

# 加载spaCy模型
nlp = spacy.load("en_core_web_sm")

class PDFUnifiedProcessor:
    """
    统一的PDF处理类，整合所有PDF处理功能
    """
    
    def __init__(self):
        self.section_keywords = {
            'abstract': ['abstract'],
            'conclusion': ['conclusion', 'conclusions', 'summary', 'final remarks'],
            'results': ['results', 'findings', 'data', 'experimental results'],
            'introduction': ['introduction', 'background'],
            'methods': ['methods', 'methodology', 'experimental', 'procedure']
        }
    
    def process_page_text(self, page_text, max_tokens=200):
        """
        使用spaCy处理页面文本，按句子分割
        """
        doc = nlp(page_text)
        parts = []
        tokens_count = 0
        content = ""
        
        for sent in doc.sents:
            sent_tokens = len(list(sent))
            if tokens_count + sent_tokens > max_tokens and content:
                parts.append((content.strip(), tokens_count))
                content = ""
                tokens_count = 0
            content += sent.text + " "
            tokens_count += sent_tokens
        
        if content:
            parts.append((content.strip(), tokens_count))
        return parts
    
    def extract_basic_text(self, pdf_path):
        """
        基础文本提取（替代原PDF_to_TXT.py功能）
        """
        doc = fitz.open(pdf_path)
        all_text = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text = page.get_text()
            if page_text:
                contents = self.process_page_text(page_text)
                for content, _ in contents:
                    all_text.append(content)
        
        doc.close()
        return all_text
    
    def extract_structured_sections(self, pdf_path):
        """
        结构化章节提取（替代PDF_Structured_Parser.py功能）
        """
        doc = fitz.open(pdf_path)
        sections = {
            'abstract': [],
            'conclusion': [],
            'results': [],
            'introduction': [],
            'methods': [],
            'other': []
        }
        
        current_section = 'other'
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            
            # 按段落分割
            paragraphs = text.split('\n\n')
            
            for paragraph in paragraphs:
                paragraph = paragraph.strip()
                if not paragraph:
                    continue
                    
                # 检查段落是否包含章节标题
                paragraph_lower = paragraph.lower()
                
                # 检查是否是新的章节
                for section, keywords in self.section_keywords.items():
                    for keyword in keywords:
                        if keyword in paragraph_lower and len(paragraph) < 200:  # 可能是标题
                            current_section = section
                            break
                
                # 将段落添加到对应章节
                if current_section in sections:
                    sections[current_section].append(paragraph)
                else:
                    sections['other'].append(paragraph)
        
        doc.close()
        return sections
    
    def save_basic_text(self, pdf_path, output_dir):
        """
        保存基础文本（兼容原PDF_to_TXT.py接口）
        """
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        specific_output_dir = os.path.join(output_dir, base_name)
        os.makedirs(specific_output_dir, exist_ok=True)
        
        output_txt_filename = os.path.join(specific_output_dir, f"{base_name}.txt")
        all_text = self.extract_basic_text(pdf_path)
        
        with open(output_txt_filename, 'w', encoding='utf-8') as output_file:
            for content in all_text:
                output_file.write(content + '\n\n')
        
        return output_txt_filename
    
    def save_structured_sections(self, sections, output_dir, base_name):
        """
        保存结构化章节
        """
        specific_output_dir = os.path.join(output_dir, base_name)
        os.makedirs(specific_output_dir, exist_ok=True)
        
        output_files = {}
        
        for section_name, paragraphs in sections.items():
            if paragraphs:  # 只保存非空章节
                output_file = os.path.join(specific_output_dir, f"{base_name}_{section_name}.txt")
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    for paragraph in paragraphs:
                        f.write(paragraph + '\n\n')
                
                output_files[section_name] = output_file
                print(f"✅ {section_name} 章节: {len(paragraphs)} 个段落 -> {output_file}")
        
        return output_files
    
    def create_priority_content(self, sections, output_dir, base_name):
        """
        创建重点内容文件（摘要+结论+结果）
        """
        priority_sections = ['abstract', 'conclusion', 'results']
        priority_content = []
        
        for section in priority_sections:
            if section in sections and sections[section]:
                priority_content.extend(sections[section])
        
        if priority_content:
            specific_output_dir = os.path.join(output_dir, base_name)
            priority_file = os.path.join(specific_output_dir, f"{base_name}_priority.txt")
            
            with open(priority_file, 'w', encoding='utf-8') as f:
                for paragraph in priority_content:
                    f.write(paragraph + '\n\n')
            
            print(f"✅ 重点内容 (摘要+结论+结果): {len(priority_content)} 个段落 -> {priority_file}")
            return priority_file
        
        return None
    
    def process_pdf_comprehensive(self, pdf_path, output_dir, mode='comprehensive'):
        """
        综合PDF处理函数
        mode: 'basic' - 基础文本提取
              'structured' - 结构化提取
              'comprehensive' - 综合处理
        """
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        results = {}
        
        if mode in ['basic', 'comprehensive']:
            # 基础文本提取
            basic_file = self.save_basic_text(pdf_path, output_dir)
            results['basic'] = basic_file
        
        if mode in ['structured', 'comprehensive']:
            # 结构化提取
            sections = self.extract_structured_sections(pdf_path)
            structured_files = self.save_structured_sections(sections, output_dir, base_name)
            results.update(structured_files)
            
            # 创建重点内容
            priority_file = self.create_priority_content(sections, output_dir, base_name)
            if priority_file:
                results['priority'] = priority_file
        
        return results

# 兼容性函数 - 保持与原代码的接口一致
def save_contents_to_specific_folders(pdf_files, base_output_dir):
    """
    兼容原PDF_to_TXT.py的接口
    """
    processor = PDFUnifiedProcessor()
    output_paths = []
    
    for pdf in pdf_files:
        output_file = processor.save_basic_text(pdf, base_output_dir)
        output_paths.append(output_file)
    
    return output_paths

def process_pdf_with_structure(pdf_path, output_dir):
    """
    兼容原PDF_Structured_Parser.py的接口
    """
    processor = PDFUnifiedProcessor()
    return processor.process_pdf_comprehensive(pdf_path, output_dir, mode='structured')

def process_multiple_pdfs_with_structure(pdf_files, base_output_dir):
    """
    兼容原PDF_Structured_Parser.py的接口
    """
    processor = PDFUnifiedProcessor()
    all_output_files = []
    
    for pdf_path in pdf_files:
        output_files = processor.process_pdf_comprehensive(pdf_path, base_output_dir, mode='structured')
        all_output_files.append(output_files)
    
    return all_output_files
