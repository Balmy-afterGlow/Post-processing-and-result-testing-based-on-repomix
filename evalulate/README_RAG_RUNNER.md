# RAG评估运行器使用说明

## 概述

`run.py` 是一个用于运行RAG（Retrieval-Augmented Generation）评估的主要脚本。它支持单个问题评估和批量评估两种模式，能够调用DeepSeek-V3 LLM模型，并使用三种不同的策略（basic、enhanced、compressed）来处理问题。

## 功能特性

- ✅ **单个问题评估**: 根据问题ID运行特定问题的评估
- ✅ **批量评估**: 批量处理多个问题
- ✅ **三种策略**: basic、enhanced、compressed
- ✅ **智能重试**: API调用失败时自动重试
- ✅ **结果保存**: 自动保存JSON格式的评估结果
- ✅ **进度跟踪**: 实时显示处理进度和统计信息
- ✅ **错误处理**: 完善的错误处理和日志记录

## 环境准备

### 1. 安装依赖

```bash
pip install openai
```

### 2. 设置API密钥

```bash
export DEEPSEEK_API_KEY=your_deepseek_api_key
```

### 3. 确保文件结构

```
evalulate/
├── run.py                          # 主运行脚本
├── example_usage.py                # 使用示例
├── datasets/
│   └── issues.json                 # 问题数据文件
├── utils/
│   ├── context_composer.py         # 上下文组合器
│   └── get_json_from_ai.py        # AI响应JSON提取
└── results/
    └── based_on_rag/
        ├── basic/                  # basic策略结果
        ├── enhanced/               # enhanced策略结果
        └── compressed/             # compressed策略结果
```

### 4. RAG上下文目录

确保存在RAG上下文文件：
```
../rag/relevant_context/
├── repo_name1/
│   ├── issue_id1/
│   │   └── rag_comparison_results.json
│   └── issue_id2/
│       └── rag_comparison_results.json
└── repo_name2/
    └── issue_id3/
        └── rag_comparison_results.json
```

## 使用方法

### 1. 命令行界面

#### 单个问题评估
```bash
# 运行指定问题ID的评估
python run.py --question-id 1

# 指定API密钥
python run.py --question-id 1 --api-key your_api_key
```

#### 批量评估
```bash
# 评估所有问题
python run.py

# 评估前10个问题
python run.py --start 0 --max 10

# 从第5个问题开始评估
python run.py --start 5

# 从第10个问题开始，最多评估20个
python run.py --start 10 --max 20
```

### 2. 程序化使用

```python
from run import RAGEvaluationRunner

# 初始化评估器
runner = RAGEvaluationRunner(
    issues_file="./datasets/issues.json",
    rag_context_dir="../rag/relevant_context",
    output_dir="./results/based_on_rag",
    api_key="your_api_key"
)

# 运行单个问题
result = runner.run_single_question("1")

# 批量运行
runner.run_evaluation(start_index=0, max_issues=5)
```

### 3. 使用示例脚本

```bash
# 检查环境
python example_usage.py --mode check

# 显示使用示例
python example_usage.py --mode usage

# 运行单个问题示例
python example_usage.py --mode single

# 运行批量评估示例
python example_usage.py --mode batch
```

## 输出结果

### 1. 结果文件结构

```
results/based_on_rag/
├── basic/
│   ├── 1.json
│   ├── 2.json
│   └── ...
├── enhanced/
│   ├── 1.json
│   ├── 2.json
│   └── ...
├── compressed/
│   ├── 1.json
│   ├── 2.json
│   └── ...
└── evaluation_statistics.json
```

### 2. 单个结果文件格式

每个问题的结果文件包含：
```json
{
  "analysis": "分析内容",
  "solution": "解决方案",
  "confidence": 0.85,
  "reasoning": "推理过程",
  "additional_info": {}
}
```

### 3. 统计文件格式

`evaluation_statistics.json` 包含：
```json
{
  "total_issues": 100,
  "processed_issues": 95,
  "failed_issues": 5,
  "strategy_stats": {
    "basic": {"success": 90, "failed": 5},
    "enhanced": {"success": 85, "failed": 10},
    "compressed": {"success": 88, "failed": 7}
  },
  "processing_errors": [...],
  "start_time": "2024-01-01T10:00:00",
  "end_time": "2024-01-01T11:30:00",
  "total_processing_time": 5400.0
}
```

## 配置选项

### RAGEvaluationRunner 参数

- `issues_file`: issues.json文件路径 (默认: `"./datasets/issues.json"`)
- `rag_context_dir`: RAG上下文目录 (默认: `"../rag/relevant_context"`)
- `output_dir`: 输出结果目录 (默认: `"./results/based_on_rag"`)
- `api_key`: DeepSeek API密钥 (默认: 从环境变量读取)
- `model_name`: 模型名称 (默认: `"deepseek-chat"`)
- `api_base`: API基础URL (默认: `"https://api.deepseek.com"`)

### API调用参数

- `temperature`: 0.1 (较低的随机性)
- `max_tokens`: 1000 (最大输出token数)
- `max_retries`: 3 (最大重试次数)

## 日志和监控

### 1. 日志文件

- **文件**: `rag_evaluation.log`
- **级别**: INFO及以上
- **格式**: `时间 - 级别 - 消息`

### 2. 控制台输出

实时显示：
- 当前处理的问题信息
- 进度统计 (每10个问题)
- 错误和警告信息
- 最终评估报告

### 3. 进度监控

```
进度: 15/100 (总体: 25/1000)
处理问题: 25 - Issue title...
问题 25 策略 basic: 处理成功
问题 25 策略 enhanced: 处理成功
问题 25 策略 compressed: API调用失败

==========================================
进度统计:
已处理问题: 25
basic: 成功 24, 失败 1 (成功率: 96.0%)
enhanced: 成功 23, 失败 2 (成功率: 92.0%)
compressed: 成功 22, 失败 3 (成功率: 88.0%)
==========================================
```

## 错误处理

### 1. 常见错误

- **API密钥未设置**: 设置 `DEEPSEEK_API_KEY` 环境变量
- **文件不存在**: 检查 `issues.json` 和 RAG上下文文件
- **API调用失败**: 检查网络连接和API配额
- **JSON解析失败**: AI响应格式不正确，会自动重试

### 2. 恢复机制

- **断点续传**: 已处理的问题会被跳过
- **重试机制**: API调用失败时自动重试
- **错误记录**: 所有错误都记录在日志和统计文件中

## 性能优化

### 1. 批处理建议

- 小批量测试: `--max 10`
- 分段处理: 使用 `--start` 和 `--max` 参数
- 监控API配额: 注意API调用限制

### 2. 并发控制

- 当前版本为顺序处理，避免API限制
- 每次API调用后延时1秒
- 失败重试使用指数退避策略

## 故障排除

### 1. 环境检查

```bash
python example_usage.py --mode check
```

### 2. 常见问题

**Q: 找不到RAG上下文文件**
A: 检查目录结构，确保 `../rag/relevant_context/repo_name/issue_id/rag_comparison_results.json` 存在

**Q: API调用频繁失败**
A: 检查API密钥、网络连接和API配额限制

**Q: JSON解析失败**
A: 检查AI响应格式，可能需要调整prompt或重试

**Q: 内存不足**
A: 减少 `max_issues` 参数，分批处理

### 3. 调试模式

修改日志级别为DEBUG：
```python
logging.basicConfig(level=logging.DEBUG)
```

## 扩展功能

### 1. 自定义策略

在 `utils/context_composer.py` 中添加新的策略类型。

### 2. 不同LLM模型

修改 `model_name` 和 `api_base` 参数支持其他模型。

### 3. 并发处理

可以实现多线程版本，但需要注意API限制。

## 联系和支持

如有问题或建议，请查看：
- 日志文件: `rag_evaluation.log`
- 统计文件: `results/based_on_rag/evaluation_statistics.json`
- 示例脚本: `example_usage.py`
