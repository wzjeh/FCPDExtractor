# 🔄 更新摘要 - 分阶段模型自定义功能

## 📅 更新时间
2025年10月17日

## 🎯 更新内容

为 FCPDExtractor 添加了**分阶段自定义LLM模型**的功能，让你可以为不同的处理步骤选择最合适的模型。

## ✨ 主要功能

### 1. 模型配置中心（Notebook Cell 7）

在 `OSSExtractor_Debug.ipynb` 中新增了一个专门的模型配置单元格，支持以下配置：

```python
# 步骤4: LLM内容过滤
MODEL_FILTER = 'nous-hermes-llama2-13b.Q4_0.gguf'

# 步骤5-抽象: 文本抽象
MODEL_ABSTRACT = 'nous-hermes-llama2-13b.Q4_0.gguf'

# 步骤5-总结: 参数总结
MODEL_SUMMARIZE = 'meta-llama-3.1-8b-instruct-q4_k_m-2.gguf'
STRICT_MODE = True

# 相似度筛选段落数
TOP_N_PARAGRAPHS = 10
```

### 2. 自动配置应用

配置会通过环境变量自动传递到后续处理步骤：
- `FCPD_TOP_N` → 控制嵌入相似度筛选的段落数
- `FCPD_STRICT_MODEL_NAME` → 严格模式下使用的模型名称

### 3. 实时模型提示

运行步骤4和5时，会显示当前使用的模型：

```
🚀 步骤 4/5: LLM内容过滤...
📌 使用模型: nous-hermes-llama2-13b.Q4_0.gguf

🚀 步骤 5/5: 抽象和总结...
📌 抽象模型: nous-hermes-llama2-13b.Q4_0.gguf
📌 总结模型: meta-llama-3.1-8b-instruct-q4_k_m-2.gguf (严格模式: True)
```

## 📝 修改的文件

### 1. `Text Extraction/Embedding_and_Similarity.py`

- 修改 `select_top_neighbors()` 函数，支持从环境变量 `FCPD_TOP_N` 读取配置
- 默认值保持为 10，确保向后兼容

```python
def select_top_neighbors(df):
    # 从环境变量读取 Top-N 配置，默认为 10
    top_n = int(os.getenv('FCPD_TOP_N', '10'))
    df = df.sort_values('similarity', ascending=False)
    top_neighbors = df.head(top_n)
    return top_neighbors
```

### 2. `Text Extraction/Unified_Text_Processor.py`

#### 更新的函数签名：

**`process_text_file_for_filter`**
```python
def process_text_file_for_filter(file_path, model_name='nous-hermes-llama2-13b.Q4_0.gguf'):
    processor = UnifiedTextProcessor(model_name=model_name)
    result = processor.process_text_file_comprehensive(file_path, mode='filter')
    return list(result.values())[0]
```

**`process_text_file_for_abstract`**
```python
def process_text_file_for_abstract(file_path, model_name='nous-hermes-llama2-13b.Q4_0.gguf'):
    processor = UnifiedTextProcessor(model_name=model_name)
    result = processor.process_text_file_comprehensive(file_path, mode='abstract')
    return list(result.values())[0]
```

**`process_text_file_for_summerized`**
```python
def process_text_file_for_summerized(file_path, model_name='meta-llama-3.1-8b-instruct-q4_k_m-2.gguf', strict=True):
    processor = UnifiedTextProcessor(model_name=model_name, strict=strict)
    result = processor.process_text_file_comprehensive(file_path, mode='summarize')
    return list(result.values())[0]
```

### 3. `OSSExtractor_Debug.ipynb`

#### 新增 Cell 6（Markdown）
模型配置说明文档

#### 新增 Cell 7（Python）
模型配置中心，包含：
- 模型选择变量
- 配置参数（Top-N, Strict Mode）
- 配置确认输出
- 环境变量设置

#### 更新 Cell 24（步骤4）
```python
# 使用配置的过滤模型
filter_file_path = process_text_file_for_filter(embedding_file, model_name=MODEL_FILTER)
```

#### 更新 Cell 27
简化为模块重载逻辑

#### 更新 Cell 29（步骤5）
```python
# 使用配置的抽象模型
abstract_file_path = process_text_file_for_abstract(filter_file, model_name=MODEL_ABSTRACT)

# 使用配置的总结模型
summarized_file_path = process_text_file_for_summerized(
    abstract_file_path, 
    model_name=MODEL_SUMMARIZE, 
    strict=STRICT_MODE
)
```

## 📚 新增文档

### `MODEL_CONFIG_GUIDE.md`
详细的模型配置使用指南，包含：
- 功能说明
- 配置方式
- 可用模型列表
- 推荐配置方案
- 参数说明
- 故障排查

## 🔧 向后兼容性

✅ **完全向后兼容**

- 所有函数都保留了默认参数值
- 不传入 `model_name` 参数时，使用默认模型
- 现有脚本和代码无需修改即可继续运行

## 🎯 使用场景

### 场景1：快速批量处理
使用轻量级模型提高速度：
```python
MODEL_FILTER = 'phi-3-mini-4k-instruct.Q4_0.gguf'
MODEL_ABSTRACT = 'nous-hermes-llama2-13b.Q4_0.gguf'
MODEL_SUMMARIZE = 'nous-hermes-llama2-13b.Q4_0.gguf'
```

### 场景2：高质量精细提取
使用大模型提升质量：
```python
MODEL_FILTER = 'nous-hermes-llama2-13b.Q4_0.gguf'
MODEL_ABSTRACT = 'mistral-7b-instruct-v0.1.Q4_0.gguf'
MODEL_SUMMARIZE = 'gpt-oss-20b-Q4_K_M.gguf'
```

### 场景3：内存受限环境
全程使用小模型：
```python
MODEL_FILTER = 'phi-3-mini-4k-instruct.Q4_0.gguf'
MODEL_ABSTRACT = 'phi-3-mini-4k-instruct.Q4_0.gguf'
MODEL_SUMMARIZE = 'mistral-7b-instruct-v0.1.Q4_0.gguf'
```

## ⚠️ 注意事项

1. **模块重载**：修改配置后必须运行 Cell 27 重载模块
2. **模型文件**：确保指定的 GGUF 文件存在于 `models/` 目录
3. **内存管理**：大模型需要更多 RAM，注意系统资源
4. **严格模式**：`STRICT_MODE=True` 时模型加载失败会报错，适合生产环境

## 🚀 快速开始

1. 打开 `OSSExtractor_Debug.ipynb`
2. 运行到 Cell 7，根据需要修改模型配置
3. 重新运行 Cell 7 应用配置
4. 运行 Cell 27 重载模块
5. 继续运行步骤4、5进行处理

## 📖 详细文档

完整的使用说明请参考：`MODEL_CONFIG_GUIDE.md`

## 🎉 总结

这次更新为 FCPDExtractor 带来了更高的灵活性和可控性，你现在可以：

- ✅ 为不同步骤选择最合适的模型
- ✅ 根据硬件资源调整模型大小
- ✅ 根据任务需求平衡速度和质量
- ✅ 动态调整相似度筛选的段落数量

**不改变任何现有功能，完全向后兼容，所有默认行为保持不变。**

