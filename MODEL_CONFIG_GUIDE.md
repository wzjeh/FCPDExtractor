# 📖 模型配置指南

## 🎯 功能说明

从现在开始，你可以为 FCPDExtractor 的不同处理阶段自定义使用的 LLM 模型，实现更灵活的提取策略。

## 🎛️ 配置方式

在 `OSSExtractor_Debug.ipynb` 中的 **Cell 7（模型配置单元格）** 里，你可以修改以下变量：

```python
# 步骤4: LLM内容过滤 - 使用的模型
MODEL_FILTER = 'nous-hermes-llama2-13b.Q4_0.gguf'

# 步骤5-抽象: 文本抽象 - 使用的模型
MODEL_ABSTRACT = 'nous-hermes-llama2-13b.Q4_0.gguf'

# 步骤5-总结: 参数总结 - 使用的模型
MODEL_SUMMARIZE = 'meta-llama-3.1-8b-instruct-q4_k_m-2.gguf'
STRICT_MODE = True  # 是否严格使用指定模型（不回退）

# 其他配置
TOP_N_PARAGRAPHS = 10  # 嵌入相似度筛选保留的段落数
```

## 📦 可用模型

确保以下 GGUF 模型文件在 `models/` 目录下：

| 模型文件名 | 大小 | 特点 | 推荐用途 |
|-----------|------|------|---------|
| `nous-hermes-llama2-13b.Q4_0.gguf` | ~7GB | 稳定快速，综合性能好 | 默认全流程 |
| `gpt-oss-20b-Q4_K_M.gguf` | ~12GB | 容量大，理解能力强 | 复杂总结 |
| `meta-llama-3.1-8b-instruct-q4_k_m-2.gguf` | ~5GB | Meta官方，指令遵循好 | 参数提取 |
| `mistral-7b-instruct-v0.1.Q4_0.gguf` | ~4GB | 轻量级，速度快 | 快速过滤/抽象 |
| `phi-3-mini-4k-instruct.Q4_0.gguf` | ~2GB | 超轻量，速度最快 | 简单过滤 |

## 🚀 使用流程

### 1. 修改配置

在 Notebook 的 Cell 7 中修改模型变量，例如：

```python
# 使用小模型快速过滤
MODEL_FILTER = 'phi-3-mini-4k-instruct.Q4_0.gguf'

# 使用中等模型进行抽象
MODEL_ABSTRACT = 'mistral-7b-instruct-v0.1.Q4_0.gguf'

# 使用大模型提取精准参数
MODEL_SUMMARIZE = 'gpt-oss-20b-Q4_K_M.gguf'
STRICT_MODE = True
```

### 2. 重新运行配置单元格

点击运行 Cell 7，你会看到确认信息：

```
====================================================================
🎛️  当前模型配置
====================================================================
📌 步骤4 (内容过滤):  phi-3-mini-4k-instruct.Q4_0.gguf
📌 步骤5 (文本抽象):  mistral-7b-instruct-v0.1.Q4_0.gguf
📌 步骤5 (参数总结):  gpt-oss-20b-Q4_K_M.gguf (严格模式: True)
📌 相似度Top-N:      10 个段落
====================================================================
```

### 3. 重载模块（重要！）

运行 Cell 27 强制重载 Python 模块，确保使用最新代码：

```python
import importlib
import Unified_Text_Processor as UTP
importlib.reload(UTP)
from Unified_Text_Processor import (
    process_text_file_for_filter,
    process_text_file_for_abstract,
    process_text_file_for_summerized
)
print("✅ 模块已重载，使用最新代码")
```

### 4. 运行处理步骤

依次运行步骤4、5，系统会自动使用你配置的模型：

- **步骤4**：会显示 `📌 使用模型: phi-3-mini-4k-instruct.Q4_0.gguf`
- **步骤5**：会显示 `📌 抽象模型: mistral-7b-instruct-v0.1.Q4_0.gguf` 和 `📌 总结模型: gpt-oss-20b-Q4_K_M.gguf`

## 💡 推荐配置方案

### 方案A：速度优先

适合快速测试或大批量处理：

```python
MODEL_FILTER = 'nous-hermes-llama2-13b.Q4_0.gguf'
MODEL_ABSTRACT = 'nous-hermes-llama2-13b.Q4_0.gguf'
MODEL_SUMMARIZE = 'nous-hermes-llama2-13b.Q4_0.gguf'
TOP_N_PARAGRAPHS = 10
```

### 方案B：质量优先

适合重要文献的精细提取：

```python
MODEL_FILTER = 'nous-hermes-llama2-13b.Q4_0.gguf'
MODEL_ABSTRACT = 'mistral-7b-instruct-v0.1.Q4_0.gguf'
MODEL_SUMMARIZE = 'gpt-oss-20b-Q4_K_M.gguf'
STRICT_MODE = True
TOP_N_PARAGRAPHS = 20
```

### 方案C：平衡模式（推荐）

速度和质量的平衡：

```python
MODEL_FILTER = 'nous-hermes-llama2-13b.Q4_0.gguf'
MODEL_ABSTRACT = 'nous-hermes-llama2-13b.Q4_0.gguf'
MODEL_SUMMARIZE = 'meta-llama-3.1-8b-instruct-q4_k_m-2.gguf'
STRICT_MODE = True
TOP_N_PARAGRAPHS = 10
```

## ⚙️ 参数说明

### `STRICT_MODE`

- `True`：严格使用指定模型，加载失败则报错（推荐用于生产环境）
- `False`：加载失败时自动回退到备用模型（适合测试）

### `TOP_N_PARAGRAPHS`

控制嵌入相似度筛选保留的段落数：

- `10`（默认）：适合短篇论文或快速提取
- `20-30`：适合长篇综述或需要更全面的信息
- `5`：适合非常聚焦的快速筛选

## ❗ 注意事项

1. **模型文件存在性**：确保你指定的 GGUF 文件在 `models/` 目录下
2. **内存限制**：大模型（如 20B）需要更多 RAM，注意系统资源
3. **模块重载**：修改配置后务必运行 Cell 27 重载模块
4. **文件清理**：如果要重新处理，先删除之前的中间文件（`_Filtered.txt`, `_Abstract.txt` 等）

## 🔧 故障排查

### 问题：模型加载失败

```
❌ 严格模式加载失败: xxx.gguf @ 绝对路径 -> Model file does not exist
```

**解决方法**：
1. 检查 `models/` 目录下是否有该文件
2. 确认文件名拼写正确（区分大小写）
3. 如果是新下载的模型，确保文件完整（检查文件大小）

### 问题：配置不生效

**解决方法**：
1. 重新运行 Cell 7（模型配置单元格）
2. **必须** 运行 Cell 27（模块重载单元格）
3. 或者重启 Jupyter Kernel

### 问题：内存不足

```
llama_new_context_with_model: failed to initialize Metal backend
```

**解决方法**：
1. 换用更小的模型（如 Phi-3 或 Mistral-7B）
2. 减少 `TOP_N_PARAGRAPHS` 参数
3. 关闭其他占用内存的应用

## 📚 扩展阅读

如需添加新模型：

1. 下载兼容的 GGUF 模型文件（来自 HuggingFace 或 gpt4all 官方）
2. 放入 `models/` 目录
3. 在 Cell 7 的模型列表注释中添加说明
4. 修改对应的 `MODEL_*` 变量即可使用

---

**祝你使用愉快！如有问题，请查看 `Text Extraction/Unified_Text_Processor.py` 中的模型加载逻辑。**

