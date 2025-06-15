# 📊 RAG Evaluation System

基于RAG（Retrieval-Augmented Generation）的代码知识库评估系统，提供完整的问答评估流程，从AI模型运行到多维度评估分析和可视化。

## 🌟 主要功能

### 核心能力
- ✅ **RAG问答运行** - 调用DeepSeek-V3等AI模型进行代码问答
- ✅ **多策略支持** - basic、compressed、enhanced三种RAG策略
- ✅ **多维度评估** - F1、Top-K、MAP、结构完整性等指标
- ✅ **AI相似性分析** - 基于OpenAI的语义相似度评估
- ✅ **标准答案生成** - 自动生成评估基准数据
- ✅ **结果可视化** - 生成GAIA Benchmark风格的对比图表
- ✅ **批量处理** - 支持大规模问题集的自动化评估

## ⚡ 快速开始

### 1. 环境准备
```bash
# 激活conda环境
conda activate repomind

# 确保已安装项目依赖
cd path/to/project_root && pip install -r requirements.txt

# 配置API密钥
cp .env.example .env
vim .env  # 填入你的DEEPSEEK_API_KEY
```

### 2. 数据准备（确保已完成前置步骤）
```bash
# 确保已构建向量数据库
cd ../rag/ && python build_vector_db.py

# 确保已获取相关上下文（可选优化）
cd ../rag/ && python obtain_relevant_context.py

# 生成标准答案（仅首次需要）
cd utils/ && python standard_generate.py --use-ai
```

### 3. 运行RAG评估
```bash
# 处理单个问题
python run.py --question-id issue_1

# 批量处理所有问题
python run.py

# 指定处理范围
python run.py --start 10 --max 50
```

### 4. 评估结果分析
```bash
# 运行多维度评估
python eval.py

# 生成可视化图表
python utils/chart_generate.py

# 查看结果
ls -la results/based_on_rag/
ls -la results/charts/based_on_rag/
```

## 🛠️ 核心脚本详解

| 脚本文件 | 主要功能 | 使用场景 | 输出 |
|---------|---------|---------|------|
| `run.py` | RAG问答运行器 | 🚀 AI模型调用和结果生成 | JSON格式AI回答 |
| `eval.py` | 多维度评估分析 | 📊 性能指标计算和比较 | 评估报告和指标 |
| `utils/standard_generate.py` | 标准答案生成 | 🎯 基准数据创建 | Gold standard JSON |
| `utils/chart_generate.py` | 结果可视化 | 📈 图表和报告生成 | PNG/PDF图表 |
| `utils/context_composer.py` | 上下文组装 | 🔧 提示词构建 | - |

## 📁 目录结构详解

```
evalulate/
├── README.md                    # 📄 本说明文档
├── .env                        # 🔐 环境变量配置（需创建）
├── .env.example                # 📋 环境变量模板
├── run.py                      # 🚀 RAG问答运行器
├── eval.py                     # 📊 多维度评估脚本
├── configs/                    # ⚙️ 配置文件目录
│   └── prompts.py              # 💬 提示词配置 🔍
├── datasets/                   # 📂 数据集目录
│   └── issues.json             # 🎯 测试问题集 🔍
├── results/                    # 📈 结果存储目录
│   ├── based_on_rag/           # 🤖 RAG评估结果 🔍
│   ├── based_on_gpt/           # 🧠 GPT评估结果
│   ├── gold_standard/          # ✅ 标准答案 🔍
│   └── charts/                 # 📊 可视化图表
│       └── based_on_rag/       # 📈 RAG结果图表 🔍
└── utils/                      # 🔧 工具函数目录
    ├── context_composer.py     # 🔗 上下文组装器
    ├── get_json_from_ai.py     # 🔍 AI响应JSON提取
    ├── standard_generate.py    # 🎯 标准答案生成器
    ├── chart_generate.py       # 📊 图表生成器
    └── datasets_generate.py    # 📂 数据集生成器
```

## 🔧 详细使用指南

### 1. RAG问答运行器 (`run.py`)

#### 主要功能
- 读取 `datasets/issues.json` 中的测试问题
- 为每个问题调用RAG系统获取相关上下文
- 使用DeepSeek-V3等AI模型生成答案
- 支持三种RAG策略：basic、compressed、enhanced

#### 配置选项
```python
RAGEvaluationRunner(
    issues_file="./datasets/issues.json",        # 问题数据文件
    rag_context_dir="../rag/relevant_context",   # RAG上下文目录
    output_dir="./results/based_on_rag",         # 输出结果目录
    api_key=None,                                 # API密钥
    model_name="deepseek-chat",                   # 模型名称
    api_base="https://api.deepseek.com",         # API基础URL
)
```

#### 使用示例
```bash
# 处理单个问题（推荐调试时使用）
python run.py --question-id issue_1

# 批量处理全部问题
python run.py

# 处理指定范围的问题
python run.py --start 0 --max 10

# 使用自定义API密钥
python run.py --api-key your_deepseek_key

# 查看处理日志
tail -f rag_evaluation.log
```

#### 输出格式
```json
{
  "id": "issue_1",
  "question": "如何修复VS Code中的内存泄漏问题？",
  "strategies": {
    "basic": {
      "ai_response": {
        "reason": "内存泄漏通常由未清理的事件监听器引起",
        "location": ["src/vs/workbench/contrib/..."],
        "fix": "添加适当的dispose方法..."
      },
      "processing_time": 3.45,
      "context_size": 2048,
      "timestamp": "2025-06-16T10:30:45Z"
    },
    "compressed": {...},
    "enhanced": {...}
  },
  "metadata": {
    "total_processing_time": 12.34,
    "success_rate": 1.0,
    "model_used": "deepseek-chat"
  }
}
```

### 2. 多维度评估器 (`eval.py`)

#### 评估指标详解

##### 📝 文本相似性指标
- **F1 Score** (F1分数)
  - **计算方式**: 精确度(Precision)和召回率(Recall)的调和平均数
  - **取值范围**: 0.0 - 1.0，越高越好
  - **适用场景**: 评估AI回答与标准答案在词汇层面的重叠程度
  - **解读**: F1=0.8表示AI回答包含80%的关键词汇信息
  - **计算公式**: F1 = 2 × (Precision × Recall) / (Precision + Recall)

- **Cosine Similarity** (余弦相似度)
  - **计算方式**: 基于TF-IDF向量的余弦相似度
  - **取值范围**: 0.0 - 1.0，越高越好
  - **适用场景**: 评估文本在语义空间中的相似程度
  - **解读**: 0.85表示AI回答与标准答案在语义上高度相似
  - **优势**: 不受文本长度影响，关注语义相似性

- **Sequence Similarity** (序列相似度)
  - **计算方式**: 基于字符级别的最长公共子序列
  - **取值范围**: 0.0 - 1.0，越高越好
  - **适用场景**: 评估回答的字面相似程度
  - **解读**: 0.75表示75%的字符序列与标准答案匹配

##### 🔍 信息检索指标
- **Top-K Accuracy** (Top-K准确率)
  - **计算方式**: 检查标准答案是否出现在AI回答的前K个位置中
  - **取值范围**: 0.0 - 1.0，越高越好
  - **适用场景**: 评估文件位置定位的准确性
  - **解读**: Top-3=0.9表示90%的情况下正确文件在前3个推荐中
  - **实际意义**: 模拟开发者查看推荐结果的行为

- **Mean Average Precision (MAP)** (平均精度均值)
  - **计算方式**: 所有查询的平均精度的平均值
  - **取值范围**: 0.0 - 1.0，越高越好
  - **适用场景**: 综合评估排序质量
  - **解读**: MAP=0.82表示平均排序质量为82%
  - **优势**: 同时考虑精确度和排序位置

- **Location Coverage** (位置覆盖率)
  - **计算方式**: AI回答覆盖的正确文件位置数量比例
  - **取值范围**: 0.0 - 1.0，越高越好
  - **适用场景**: 评估问题定位的全面性
  - **解读**: 0.75表示AI找到了75%的相关文件位置
  - **实际意义**: 衡量AI是否能发现所有相关的代码位置

##### 🏗️ 结构完整性指标
- **JSON Completeness** (JSON完整性)
  - **计算方式**: 检查AI输出是否包含所有必需的JSON字段
  - **取值范围**: 0.0 - 1.0，越高越好
  - **适用场景**: 评估输出格式的规范性
  - **解读**: 0.95表示95%的输出包含完整的JSON结构
  - **必需字段**: reason, location, fix

- **Field Validity** (字段有效性)
  - **计算方式**: 检查每个字段内容的有效性和合理性
  - **取值范围**: 0.0 - 1.0，越高越好
  - **适用场景**: 评估输出内容的质量
  - **检查项**: 字段非空、类型正确、内容合理
  - **解读**: 0.88表示88%的字段内容符合预期格式

- **Content Quality** (内容质量)
  - **计算方式**: 综合评估回答的逻辑性、完整性和实用性
  - **取值范围**: 0.0 - 1.0，越高越好
  - **评估维度**: 
    - 逻辑连贯性 (25%)
    - 技术准确性 (35%)
    - 解决方案可行性 (40%)
  - **解读**: 0.83表示内容质量为83%，属于高质量回答

##### 🤖 AI相似性分析指标
- **Semantic Similarity** (语义相似度)
  - **计算方式**: 使用GPT/DeepSeek模型计算语义相似度
  - **取值范围**: 0.0 - 1.0，越高越好
  - **适用场景**: 深度语义理解评估
  - **解读**: 0.87表示AI回答在语义上与标准答案高度一致
  - **优势**: 能够理解同义表达和上下文含义

- **Reasoning Quality** (推理质量)
  - **计算方式**: AI评估推理过程的逻辑性和合理性
  - **取值范围**: 0.0 - 1.0，越高越好
  - **评估要素**:
    - 问题分析深度 (30%)
    - 因果关系推理 (40%)
    - 解决思路清晰度 (30%)
  - **解读**: 0.79表示推理过程质量良好

- **Solution Relevance** (解决方案相关性)
  - **计算方式**: 评估提供的解决方案与问题的匹配度
  - **取值范围**: 0.0 - 1.0，越高越好
  - **评估标准**:
    - 解决方案针对性 (40%)
    - 技术可行性 (35%)
    - 实施复杂度合理性 (25%)
  - **解读**: 0.84表示解决方案高度相关且可行

##### 📊 指标权重和综合评分
```python
# 综合评分计算公式
weighted_score = (
    f1_score * 0.15 +           # 文本相似性
    cosine_similarity * 0.15 +   # 语义相似性  
    top_k_accuracy * 0.20 +      # 位置准确性
    map_score * 0.15 +           # 排序质量
    json_completeness * 0.10 +   # 结构完整性
    ai_similarity * 0.25         # AI语义分析
)
```

##### 🎯 指标解读指南
| 分数区间 | 质量等级 | 说明 | 建议行动 |
|---------|---------|------|---------|
| 0.90+ | 🟢 优秀 | 接近人类专家水平 | 保持现有策略 |
| 0.80-0.89 | 🔵 良好 | 达到实用标准 | 微调优化 |
| 0.70-0.79 | 🟡 中等 | 基本可用，有改进空间 | 重点优化 |
| 0.60-0.69 | 🟠 偏低 | 需要显著改进 | 策略调整 |
| <0.60 | 🔴 较差 | 不建议生产使用 | 重新设计 |

##### 📈 指标趋势分析
- **横向比较**: 同一问题在不同策略下的表现差异
- **纵向分析**: 同一策略在不同问题类型上的稳定性
- **相关性分析**: 各指标间的相关关系和互补性
- **时间趋势**: 评估结果随时间的变化趋势

##### 🔧 指标优化建议
1. **F1 Score偏低**: 优化关键词提取和答案生成策略
2. **Top-K Accuracy不足**: 改进文件位置检索算法
3. **JSON Completeness问题**: 加强输出格式控制
4. **AI Similarity较低**: 调整提示词和上下文策略
5. **Overall Score波动大**: 分析不同问题类型的处理策略

#### 使用示例
```bash
# 标准评估（使用默认参数）
python eval.py

# 自定义目录
python eval.py --rag-dir ./results/based_on_rag --gold-dir ./results/gold_standard

# 评估特定策略
python eval.py --strategies basic enhanced

# 禁用AI分析（更快速度）
python eval.py --disable-ai

# 使用自定义API
python eval.py --api-key your_key --openai-base-url https://api.openai.com/v1

# 调试模式
python eval.py --log-level DEBUG
```

#### 评估报告示例
```json
{
  "overall_statistics": {
    "total_questions": 100,
    "evaluation_date": "2025-06-16T10:30:45Z",
    "strategies_evaluated": ["basic", "compressed", "enhanced"]
  },
  "strategy_results": {
    "basic": {
      "f1_score": 0.742,
      "cosine_similarity": 0.681,
      "top_k_accuracy": 0.850,
      "map_score": 0.723,
      "json_completeness": 0.980,
      "ai_similarity": 0.756
    },
    "enhanced": {
      "f1_score": 0.834,
      "cosine_similarity": 0.798,
      "top_k_accuracy": 0.920,
      "map_score": 0.867,
      "json_completeness": 0.990,
      "ai_similarity": 0.823
    }
  },
  "comparative_analysis": {
    "metric_rankings": {
      "enhanced": 0.8472,
      "compressed": 0.7891,
      "basic": 0.7234
    },
    "improvement_analysis": {
      "enhanced": {"improvement_percentage": 17.1},
      "compressed": {"improvement_percentage": 9.1}
    }
  }
}
```

### 3. 标准答案生成器 (`utils/standard_generate.py`)

#### 主要功能
- 解析 `datasets/issues.json` 中的diff信息
- 使用AI模型生成标准答案
- 保存为JSON格式的gold standard

#### 使用示例
```bash
# 进入utils目录
cd utils/

# 生成标准答案（使用AI）
python standard_generate.py --use-ai

# 使用自定义输出目录
python standard_generate.py --output-dir ../results/custom_gold --use-ai

# 使用自定义API密钥
python standard_generate.py --use-ai --api-key your_deepseek_key

# 生成占位符答案（不调用AI）
python standard_generate.py
```

### 4. 图表生成器 (`utils/chart_generate.py`)

#### 生成的图表类型
- 📊 **策略对比柱状图**: 各策略在不同指标上的表现
- 📈 **性能趋势图**: 评估指标随时间的变化
- 🎯 **雷达图**: 多维度性能综合展示
- 📉 **分布直方图**: 各指标的分布情况
- 🔄 **改进分析图**: 相对于基准策略的改进情况

#### 使用示例
```bash
# 进入utils目录
cd utils/

# 生成全部图表
python chart_generate.py

# 自定义输入输出目录
python chart_generate.py \
  --input-dir ../results/based_on_rag \
  --output-dir ../results/charts/custom

# 生成特定类型图表
python chart_generate.py --chart-types bar radar

# 自定义图表样式
python chart_generate.py --style seaborn --dpi 300
```

## 📊 重要配置文件

### 环境变量配置 (`.env`)
```bash
# DeepSeek API密钥（必需）
DEEPSEEK_API_KEY=your_deepseek_api_key_here

# API基础URL（可选）
OPENAI_BASE_URL=https://api.deepseek.com

# 获取API密钥: https://platform.deepseek.com/
```

### 提示词配置 (`configs/prompts.py`)
包含RAG系统使用的核心提示词：
- `system_prompt`: 系统角色定义
- `format_user_prompt_from_rag()`: 用户提示词模板
- 结构化JSON输出格式定义

### 测试数据集 (`datasets/issues.json`)
包含测试问题的JSON数组：
```json
[
  {
    "id": "issue_1",
    "repo": "microsoft/vscode",
    "issue_title": "Memory leak in extension host",
    "issue_body": "详细的问题描述...",
    "changed_files": ["src/vs/workbench/..."],
    "diff": "Git diff内容...",
    "base_sha": "commit_hash"
  }
]
```

## 🔍 故障排除

### 常见问题

#### 1. API连接失败
```bash
# 错误：Failed to connect to DeepSeek API
# 解决：检查API密钥和网络连接
cat .env  # 确认API密钥正确
curl -H "Authorization: Bearer $DEEPSEEK_API_KEY" https://api.deepseek.com/
```

#### 2. 上下文文件缺失
```bash
# 错误：FileNotFoundError: relevant_context not found
# 解决：确保已运行RAG预处理
cd ../rag/
python obtain_relevant_context.py
```

#### 3. 标准答案缺失
```bash
# 错误：Gold standard files not found
# 解决：生成标准答案
cd utils/
python standard_generate.py --use-ai
```

#### 4. 内存不足
```bash
# 错误：OutOfMemoryError during evaluation
# 解决：减少批处理大小或使用分段处理
python run.py --start 0 --max 10  # 分批处理
```

#### 5. 图表生成失败
```bash
# 错误：matplotlib backend error
# 解决：设置正确的显示后端
export MPLBACKEND=Agg
python utils/chart_generate.py
```

## 📈 性能监控

### 日志文件
- `rag_evaluation.log`: RAG运行日志
- `evaluation.log`: 评估过程日志
- `chart_generation.log`: 图表生成日志

### 性能指标
- **处理速度**: 平均每个问题处理时间
- **成功率**: API调用成功率
- **资源使用**: 内存和CPU使用情况
- **准确性**: 各维度评估指标

### 监控命令
```bash
# 查看实时日志
tail -f rag_evaluation.log

# 检查处理进度
ls -la results/based_on_rag/ | wc -l

# 查看评估统计
cat results/based_on_rag/evaluation_results.json | jq '.overall_statistics'

# 监控系统资源
htop  # 或 top
```

## 🚀 进阶使用

### 自定义评估指标
```python
# 在eval.py中添加自定义指标
def custom_metric(prediction, ground_truth):
    # 实现自定义评估逻辑
    return score

# 在RAGEvaluator类中注册指标
evaluator.add_custom_metric("custom", custom_metric)
```

### 批量API优化
```python
# 并行处理多个问题
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def batch_process_questions(questions):
    with ThreadPoolExecutor(max_workers=4) as executor:
        tasks = [executor.submit(process_question, q) for q in questions]
        results = [task.result() for task in tasks]
    return results
```

### 结果数据分析
```python
# 加载和分析评估结果
import pandas as pd
import json

# 读取评估结果
with open('results/based_on_rag/evaluation_results.json') as f:
    data = json.load(f)

# 创建DataFrame进行分析
df = pd.DataFrame(data['strategy_results']).T
print(df.describe())

# 生成相关性分析
correlation_matrix = df.corr()
```

**最后更新**: 2025-06-16  
**版本**: 1.0.0