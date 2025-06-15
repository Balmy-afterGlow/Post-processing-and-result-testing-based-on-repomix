# 🧠 RAG (Retrieval-Augmented Generation) System

基于向量检索的增强生成系统，用于代码知识库的智能问答。提供完整的向量数据库构建、相关上下文检索和RAG系统核心功能。

## 🌟 主要功能

### 核心能力
- ✅ **批量向量数据库构建** - 从Repomix生成的文档构建向量数据库
- ✅ **智能文档解析** - 解析Markdown文档，提取代码块和Git历史
- ✅ **语义相似度检索** - 基于问题检索最相关的代码上下文
- ✅ **多版本支持** - 支持标准版、压缩版、Git增强版文档
- ✅ **批量上下文获取** - 为大批量问题预先获取相关上下文
- ✅ **性能优化** - 文档过滤、批处理、智能采样等优化策略

## ⚡ 快速开始

### 1. 环境准备
```bash
# 激活conda环境
conda activate repomind

# 确保已安装项目依赖
cd path/to/project_root && pip install -r requirements.txt

# 验证环境
python -c "import langchain, chromadb, sentence_transformers; print('✅ 环境检查通过')"
```

### 2. 构建向量数据库（必需的第一步）
```bash
# 从Repomix生成的文档构建向量数据库
python build_vector_db.py

# 输出目录结构
# ./vector_dbs/
# ├── repository-vscode/
# │   ├── standard/           # 📊 标准版向量数据库
# │   ├── compressed/         # 🗜️ 压缩版向量数据库
# │   └── git_enhanced/       # 🔄 Git增强版向量数据库
# ├── repository-react/
# └── build_statistics.json   # 📈 构建统计信息
```

### 3. 获取相关上下文
```bash
# 为issues.json中的问题预先获取相关上下文
python obtain_relevant_context.py

# 输出目录结构
# ./relevant_context/
# ├── microsoft-vscode/
# │   ├── issue_1/
# │   │   └── rag_comparison_results.json  # 📊 RAG检索结果比较
# │   ├── issue_2/
# │   └── ...
# └── context_retrieval_statistics.json    # 📈 检索统计信息
```

## 🛠️ 核心脚本详解

| 脚本文件 | 主要功能 | 使用场景 | 输出 |
|---------|---------|---------|------|
| `rag_system.py` | RAG系统核心模块 | 🏗️ 提供基础类和方法 | 无直接输出 |
| `build_vector_db.py` | 向量数据库构建 | 🏭 首次部署/数据更新 | 向量数据库文件 |
| `obtain_relevant_context.py` | 批量上下文检索 | 🎯 预处理优化 | 上下文JSON文件 |

## 🔧 详细使用指南

### 1. 向量数据库构建

#### 脚本功能
`build_vector_db.py` 是RAG系统的基础组件，负责：
- 解析 `../repomix_md/` 目录下的Markdown文档
- 提取代码块、文件信息和Git提交历史
- 构建基于ChromaDB的向量数据库
- 支持三种文档版本的并行处理

#### 配置选项
```python
# 可在脚本中修改的优化参数
OptimizedVectorDatabaseBuilder(
    repomix_dir="../repomix_md",           # Repomix输出目录
    output_dir="./vector_dbs",             # 向量数据库输出目录
    max_documents_per_version=2000,        # 每版本最大文档数（性能优化）
    min_content_length=100,                # 最小内容长度过滤
    batch_size=100,                        # 批处理大小
    skip_large_files=False,                # 是否跳过大文件
    max_file_size_mb=5.0,                 # 最大处理文件大小
)
```

#### 使用示例
```bash
# 基本使用
python build_vector_db.py

# 查看构建日志
tail -f vector_db_build.log

# 检查构建结果
ls -la vector_dbs/
cat vector_dbs/build_statistics.json
```

#### 输出说明
```
vector_dbs/
├── repository-{repo_name}/
│   ├── standard/                    # 标准版本向量数据库
│   │   ├── chroma.sqlite3          # ChromaDB数据文件
│   │   └── chroma_collection/      # 向量集合数据
│   ├── compressed/                  # 压缩版本向量数据库
│   └── git_enhanced/               # Git增强版本向量数据库
├── build_statistics.json          # 详细构建统计
├── verification_results.json      # 验证结果
└── vector_db_build.log            # 构建日志
```

#### 统计信息示例
```json
{
  "processed_repos": 15,
  "failed_repos": 0,
  "skipped_repos": 2,
  "total_documents": 28450,
  "filtered_documents": 3210,
  "total_processing_time_seconds": 1847.2,
  "repositories": {
    "microsoft-vscode": {
      "standard": {"documents": 1892, "avg_length": 542},
      "compressed": {"documents": 1456, "avg_length": 387},
      "git_enhanced": {"documents": 1892, "avg_length": 1024}
    }
  }
}
```

### 2. 相关上下文检索

#### 脚本功能
`obtain_relevant_context.py` 提供批量上下文预检索功能：
- 从 `../evalulate/datasets/issues.json` 读取问题
- 为每个问题在对应仓库的向量数据库中检索相关上下文
- 比较三种文档版本的检索效果
- 生成结构化的上下文检索结果

#### 配置选项
```python
RelevantContextRetriever(
    vector_db_dir="./vector_dbs",              # 向量数据库目录
    issues_file="../evalulate/datasets/issues.json",  # 问题数据文件
    output_dir="./relevant_context",           # 上下文输出目录
)
```

#### 检索策略
脚本支持多种检索策略：
- **标准检索**: 基于问题原始描述进行检索
- **增强检索**: 结合问题标题和描述的组合检索
- **多样性检索**: 使用不同的检索参数获取多样化结果

#### 使用示例
```bash
# 执行批量上下文检索
python obtain_relevant_context.py

# 查看检索日志
tail -f obtain_relevant_context.log

# 检查某个问题的检索结果
cat relevant_context/microsoft-vscode/issue_1/rag_comparison_results.json
```

#### 输出格式
```json
{
  "issue_id": "issue_1",
  "question": "如何在VS Code中配置自定义快捷键？",
  "repository": "microsoft/vscode",
  "retrieval_results": {
    "standard": {
      "contexts": [
        {
          "content": "// 快捷键配置相关代码...",
          "file_path": "src/vs/workbench/contrib/keybinding/...",
          "similarity_score": 0.847,
          "metadata": {...}
        }
      ],
      "total_retrieved": 5,
      "avg_similarity": 0.782
    },
    "compressed": {...},
    "git_enhanced": {...}
  },
  "retrieval_timestamp": "2025-06-16T10:30:45",
  "processing_time_seconds": 2.34
}
```

### 3. RAG系统核心模块

#### 主要组件
`rag_system.py` 提供以下核心类：

##### MarkdownParser
```python
# Markdown文档解析器
parser = MarkdownParser()
code_blocks = parser.parse_markdown(md_content)
```

##### DocumentProcessor
```python
# 文档处理器
processor = DocumentProcessor(max_chunk_size=1000)
documents = processor.process_code_blocks(code_blocks)
```

##### LangChainRAGSystem
```python
# RAG系统核心
rag_system = LangChainRAGSystem(collection_name="repo_name")
rag_system.add_documents(documents)
results = rag_system.query("问题描述", k=5)
```

#### 关键特性
- **离线运行**: 配置为使用本地缓存，避免网络请求
- **多模型支持**: 支持不同的嵌入模型
- **灵活配置**: 可调整检索参数和文档处理策略
- **Git集成**: 支持包含Git提交历史的增强检索

## 📊 性能优化

### 构建优化
- **文档过滤**: 过滤小于100字符的文档
- **批处理**: 100文档为一批进行向量化
- **内存管理**: 限制每版本最大文档数为2000
- **并行处理**: 多版本并行构建

### 检索优化
- **上下文缓存**: 预先检索并缓存常见问题的上下文
- **相似度阈值**: 设置最低相似度要求
- **结果去重**: 避免返回重复的代码片段

## 🔍 故障排除

### 常见问题

#### 1. 依赖检查失败
```bash
# 错误：ImportError: rag_system module not found
# 解决：确保在rag目录下运行脚本
cd rag/
python build_vector_db.py
```

#### 2. 向量数据库构建失败
```bash
# 错误：No such file or directory: '../repomix_md'
# 解决：确保已运行repomix生成文档
cd ../scripts/
python batch_generate_md.py
```

#### 3. 内存不足
```bash
# 错误：OutOfMemoryError during embedding
# 解决：减少max_documents_per_version参数
# 在build_vector_db.py中修改：
max_documents_per_version=1000  # 从2000降到1000
```

#### 4. ChromaDB锁定错误
```bash
# 错误：sqlite3.OperationalError: database is locked
# 解决：清理现有数据库
rm -rf vector_dbs/*/
python build_vector_db.py
```

## 📈 监控和维护

### 日志文件
- `vector_db_build.log`: 向量数据库构建日志
- `obtain_relevant_context.log`: 上下文检索日志

### 统计文件
- `build_statistics.json`: 构建过程详细统计
- `verification_results.json`: 数据库验证结果
- `context_retrieval_statistics.json`: 检索性能统计

### 定期维护
```bash
# 清理旧的向量数据库
rm -rf vector_dbs/*/

# 重新构建（当源文档更新时）
python build_vector_db.py

# 验证数据库完整性
python -c "
from rag_system import LangChainRAGSystem
import os
for db in os.listdir('vector_dbs/'):
    if os.path.isdir(f'vector_dbs/{db}/standard'):
        rag = LangChainRAGSystem(f'{db}_standard')
        print(f'✅ {db}: {rag.get_collection_count()} documents')
"
```

## 🚀 进阶使用

### 自定义嵌入模型
```python
# 在rag_system.py中修改模型
LangChainRAGSystem(
    collection_name="repo_name",
    model_name="all-mpnet-base-v2"  # 更高精度的模型
)
```

### 批量API调用
```python
# 批量处理多个问题
from rag_system import LangChainRAGSystem

rag = LangChainRAGSystem("microsoft-vscode_standard")
questions = ["问题1", "问题2", "问题3"]
results = [rag.query(q, k=3) for q in questions]
```

**最后更新**: 2025-06-16  
**版本**: 1.0.0