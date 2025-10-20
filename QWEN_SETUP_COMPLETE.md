# ✅ Qwen-Plus配置完成

## 已完成的所有修改

### 1. 新文件
- ✅ `core/models/qwen_llm.py` - Qwen API接口实现

### 2. 修改的文件
- ✅ `config.yaml` - 添加qwen配置，默认引擎改为qwen
- ✅ `main.py` - 添加qwen引擎支持
- ✅ `requirement.txt` - 添加openai依赖
- ✅ `OSSExtractor_Debug.ipynb`:
  - Cell 0: 设置Qwen API密钥
  - Cell 1: 改为占位单元格
  - Cell 9: 添加qwen引擎分支

### 3. 依赖安装
- ✅ `pip install openai` 已完成

### 4. 目录创建
- ✅ `data/qwen/` 已创建

---

## 🚀 立即使用

### 在Notebook中测试：

1. **重启内核** (Kernel → Restart Kernel)
2. **运行Cell 0** - 设置API密钥
   ```
   ✅ Qwen API密钥已设置，引擎: qwen
   ```
3. **跳过Cell 1** - 无需执行
4. **运行Cell 4** - 导入模块
5. **运行Cell 9** - 引擎初始化
   ```
   📌 引擎:          qwen
   📌 归档目录名:     qwen
   ✅ 处理器初始化完成
   ```
6. **依次运行步骤1-5** - 完整提取流程

### 预期输出：

**步骤5（无安全过滤错误）：**
```
📄 处理文件 1/1: Embedding_101021acsoprd7b00291_Filtered.txt
🔍 处理文件: Embedding_101021acsoprd7b00291_Filtered.txt
  ✅ 抽象完成 -> ...
🔍 处理文件: ...
  ✅ 逐段总结 -> ...
  🧩 整篇汇总 -> ...
  📊 影响因素 -> ...

📊 最终产物汇总:
  - 抽象文件: 1 个
  - 逐段总结: 1 个
  - 整篇汇总(Overall): 1 个
  - 影响因素(Impact): 1 个
```

**输出文件位置：**
- `data/qwen/101021acsoprd7b00291/`

---

## 🔄 引擎切换

### 切换到本地LLM：
```python
# Cell 0
os.environ['NB_ENGINE'] = 'local'
```

### 切换到Gemini（如需测试）：
```python
# Cell 0
os.environ['GOOGLE_API_KEY'] = 'your_key'
os.environ['NB_ENGINE'] = 'gemini'
```

### 切换到Qwen（当前默认）：
```python
# Cell 0
os.environ['QWEN_API_KEY'] = 'sk-e950e56cc74d4d89bd21f3866fa7ff51'
os.environ['NB_ENGINE'] = 'qwen'
```

每次切换后重新运行Cell 9（引擎初始化）。

---

## 📊 三个引擎对比

| 特性 | 本地LLM | Qwen-Plus | Gemini |
|------|---------|-----------|--------|
| 安全过滤 | ✅ 无 | ✅ 无 | ❌ 严格（硝化反应被拒） |
| 速度 | 慢 (~5分钟) | 快 (~1分钟) | 快 (~2分钟) |
| 成本 | 免费 | ¥0.4/M tokens | $0.075/M tokens |
| 质量 | 中等 | 高 | 高（但被拦截） |
| 离线可用 | ✅ | ❌ | ❌ |
| 推荐场景 | 离线/预算有限 | **硝化类论文** | 其他化学论文 |

---

## ✅ 验证清单

运行Notebook后检查：

- [ ] Cell 0 输出: `✅ Qwen API密钥已设置，引擎: qwen`
- [ ] Cell 9 输出: `📌 引擎: qwen`, `📌 归档目录名: qwen`
- [ ] 步骤5 无 `finish_reason=2` 错误
- [ ] `data/qwen/101021acsoprd7b00291/_Overall.txt` 包含有效JSON
- [ ] `data/qwen/101021acsoprd7b00291/_Impact_Analysis.txt` 包含因果关系表格

---

现在开始测试！

