# Post-processing and Result Testing Based on Repomix

## 项目概述

本项目是基于Repomix的后处理和结果测试框架，主要用于评估RAG（Retrieval-Augmented Generation）系统在代码知识库问答任务上的性能。项目包含完整的实验流程，从数据预处理、向量数据库构建、RAG系统运行到结果评估和可视化。

## 目录结构

```
post-processing_and_result-testing/
├── README.md                 # 项目说明文档
├── requirements.txt          # Python依赖包列表
├── evalulate/               # 评估系统核心模块
│   ├── eval.py             # 评估脚本
│   ├── run.py              # RAG系统运行器
│   ├── README_RAG_RUNNER.md # RAG运行器详细说明
│   ├── configs/            # 配置文件目录
│   │   └── prompts.py      # 提示词配置 🔍
│   ├── datasets/           # 数据集目录
│   │   └── issues.json     # 设计用例数据 🔍
│   ├── results/            # 结果存储目录
│   │   ├── gold_standard/  # 标准答案目录 🔍
│   │   ├── based_on_rag/   # RAG评估结果 🔍
│   │   ├── based_on_gpt/   # GPT评估结果
│   │   └── charts/         # 可视化图表
│   │       └── based_on_rag/ # RAG结果图表 🔍
│   └── utils/              # 工具函数
├── rag/                     # RAG系统模块
│   ├── build_vector_db.py  # 向量数据库构建
│   ├── rag_system.py       # RAG系统核心
│   ├── obtain_relevant_context.py # 相关上下文获取
│   ├── README_build_vector_db.md  # 向量数据库构建说明
│   ├── vector_dbs/         # 向量数据库存储
│   └── relevant_context/   # 相关上下文缓存
├── repomix_md/             # Repomix生成的Markdown文件
└── scripts/                # 辅助脚本
    ├── README.md           # 脚本说明
    ├── add_commit_info.py  # 添加提交信息
    └── batch_generate_md.py # 批量生成Markdown
```

## 环境配置

### 1. 创建Conda虚拟环境

```bash
# 创建名为repomind的虚拟环境，指定Python 3.12版本
conda create -n repomind python=3.12

# 激活虚拟环境
conda activate repomind
```

### 2. 安装依赖包

```bash
# 在项目根目录下安装所有依赖
pip install -r requirements.txt
```

### 3. 环境变量配置

在 `evalulate/` 目录下创建 `.env` 文件，参考 `.env.example` 配置必要的环境变量：

```bash
# 复制环境变量模板
cp evalulate/.env.example evalulate/.env

# 编辑环境变量文件，配置API密钥等
vim evalulate/.env
```

## 重要文件指南

### 📋 设计用例
- **文件位置**: `./evalulate/datasets/issues.json`
- **说明**: 包含所有测试用例的问题描述、issue url等信息

### 💬 提示词配置
- **文件位置**: `./evalulate/configs/prompts.py`
- **说明**: 定义了RAG系统使用的各种提示词模板

### ✅ 标准答案
- **文件位置**: `./evalulate/results/gold_standard/`
- **说明**: 包含每个测试用例的标准答案，用于评估系统性能

### 📊 评估指标
- **文件位置**: `./evalulate/results/based_on_rag/`
- **说明**: RAG系统的评估结果，包含各种性能指标

### 📈 可视化结果
- **文件位置**: `./evalulate/results/charts/based_on_rag/`
- **说明**: 评估结果的可视化图表，便于分析系统性能

### 🗄️ 向量数据库
- **文件位置**: `./rag/vector_dbs/`
- **说明**: 构建的向量数据库文件，用于语义检索

### 📄 相关上下文
- **文件位置**: `./rag/relevant_context/`
- **说明**: 缓存的相关上下文信息，提高检索效率

## 使用流程

### 1. 运行RAG评估
```bash
cd evalulate/

# 单个问题评估
python run.py --question_id <问题ID>

# 批量评估
python run.py --batch

# 指定策略评估
python run.py --strategy enhanced --batch
```

### 2. 评估结果分析
```bash
# 运行评估脚本
python eval.py

# 查看结果
# 数值结果: ./results/based_on_rag/
# 可视化图表: ./results/charts/based_on_rag/
```

## 主要功能模块

### RAG系统 (`rag/`)
- **向量数据库构建**: 将代码知识库转换为向量表示
- **语义检索**: 基于问题检索相关代码片段
- **上下文生成**: 生成增强的上下文信息

### 评估系统 (`evalulate/`)
- **问题处理**: 支持单个和批量问题评估
- **多策略支持**: basic、enhanced、compressed三种策略
- **性能评估**: 多维度评估指标计算
- **结果可视化**: 自动生成评估图表

### 辅助工具 (`scripts/`)
- **批量处理**: 批量生成Markdown文档
- **信息增强**: 添加Git提交信息等元数据

## 依赖说明

主要依赖包包括：
- **LangChain**: LLM应用开发框架
- **ChromaDB**: 向量数据库
- **Sentence Transformers**: 文本嵌入模型
- **OpenAI**: GPT模型API
- **Matplotlib/Seaborn**: 数据可视化
- **Scikit-learn**: 机器学习评估指标

## 注意事项

1. 确保已正确配置API密钥和环境变量
2. 首次运行需要下载预训练模型，可能需要较长时间
3. 向量数据库构建需要较大存储空间
4. 建议在GPU环境下运行以提高处理速度

## 相关文档

- [RAG运行器详细说明](./evalulate/README_RAG_RUNNER.md)
- [向量数据库构建说明](./rag/README_build_vector_db.md)
- [脚本工具说明](./scripts/README.md)
