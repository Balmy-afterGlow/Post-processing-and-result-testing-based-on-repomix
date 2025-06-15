# 📊 向量数据库批量构建工具

这个工具可以批量处理 `../repomix_md` 目录下的所有仓库，为每个包含Git增强版markdown文档的仓库构建向量数据库。

## 🚀 快速开始

### 1. 环境准备

确保已安装必需的依赖：

```bash
# 安装LangChain和相关依赖
pip install langchain langchain-huggingface langchain-community chromadb tiktoken

# 如果使用HuggingFace embeddings，可能需要
pip install sentence-transformers
```

### 2. 运行构建

```bash
cd rag
python build_vector_db.py
```

## 📁 输入输出

### 输入目录结构
脚本会自动扫描以下结构：
```
../repomix_md/
├── repository-vscode/
│   ├── repomix-output-vscode-with-git.md     # ✅ 会被处理
│   ├── repomix-output-vscode.md              # ⏭️ 跳过
│   └── ...
├── repository-react/
│   ├── repomix-output-react-with-git.md      # ✅ 会被处理
│   └── ...
└── ...
```

### 输出目录结构
```
./vector_dbs/
├── repository-vscode/
│   ├── chroma_db_basic/           # 基础版本向量数据库
│   ├── chroma_db_enhanced/        # 增强版本向量数据库
│   ├── chroma_db_compressed/      # 压缩版本向量数据库
│   └── metadata.json             # 仓库元数据
├── repository-react/
│   ├── chroma_db_basic/
│   ├── chroma_db_enhanced/
│   ├── chroma_db_compressed/
│   └── metadata.json
├── build_statistics.json         # 构建统计信息
└── verification_results.json     # 验证结果
```

## 🔧 功能特性

### 三种向量数据库版本

1. **基础版本 (basic)**: 仅包含代码内容
2. **增强版本 (enhanced)**: 代码内容 + 完整Git提交历史
3. **压缩版本 (compressed)**: 关键代码 + 简化提交信息

### 自动化特性

- ✅ **自动发现**: 扫描所有Git增强版markdown文件
- ✅ **批量处理**: 一次性处理所有仓库
- ✅ **错误恢复**: 单个仓库失败不影响其他仓库
- ✅ **进度监控**: 实时显示处理进度和状态
- ✅ **结果验证**: 自动验证构建结果的完整性

### 详细日志

- 📝 **控制台输出**: 实时显示处理状态
- 📁 **日志文件**: `vector_db_build.log` 包含详细日志
- 📊 **统计报告**: JSON格式的详细统计信息

## 📊 输出文件说明

### metadata.json (每个仓库)
```json
{
  "repo_name": "vscode",
  "source_file": "../repomix_md/repository-vscode/repomix-output-vscode-with-git.md",
  "build_time": "2025-06-14T15:30:00",
  "code_blocks_count": 156,
  "documents_count": {
    "basic": 245,
    "enhanced": 389,
    "compressed": 178
  },
  "total_commits": 523,
  "files_processed": ["src/main.ts", "src/config.js", ...]
}
```

### build_statistics.json (全局)
```json
{
  "total_repos": 25,
  "processed_repos": 23,
  "skipped_repos": 1,
  "failed_repos": 1,
  "total_documents": 18750,
  "processing_errors": [
    {
      "repo_name": "failed-repo",
      "error": "解析错误信息",
      "timestamp": "2025-06-14T15:35:00"
    }
  ]
}
```

## 🎯 使用场景

### RAG系统开发
为每个代码仓库建立独立的知识库，支持：
- 代码问答
- 功能查找
- Bug分析
- 开发历史追踪

### 多版本比较
三种不同的向量数据库版本适用于：
- **基础版**: 快速代码搜索
- **增强版**: 包含完整上下文的深度分析
- **压缩版**: 平衡性能和信息密度

## ⚠️ 注意事项

### 资源消耗
- **内存**: 大型仓库可能消耗较多内存
- **磁盘**: 每个仓库约占用100-500MB磁盘空间
- **时间**: 根据仓库大小，单个仓库处理时间1-10分钟

### 依赖要求
- **Python 3.8+**
- **足够的磁盘空间**: 建议至少5GB可用空间
- **网络连接**: 首次运行时下载embedding模型

### 错误处理
- 如果单个仓库处理失败，会继续处理其他仓库
- 所有错误信息都会记录在日志中
- 可以重新运行脚本，会自动跳过已成功的仓库

## 🔍 验证和调试

### 检查构建结果
```bash
# 查看构建统计
cat ./vector_dbs/build_statistics.json

# 查看验证结果  
cat ./vector_dbs/verification_results.json

# 检查具体仓库
ls -la ./vector_dbs/repository-vscode/
```

### 测试向量数据库
```python
from rag_system import LangChainRAGSystem

# 加载向量数据库
rag = LangChainRAGSystem("vscode_enhanced")
rag.persist_directory = "./vector_dbs/repository-vscode/chroma_db_enhanced"

# 测试搜索
results = rag.search("authentication function", k=5)
for result in results:
    print(f"文件: {result['metadata']['file_path']}")
    print(f"相似度: {result['similarity_score']:.3f}")
```

## 🚀 高级用法

### 自定义配置
可以修改脚本中的参数：

```python
# 修改输入输出目录
builder = VectorDatabaseBuilder(
    repomix_dir="../custom_repomix_md", 
    output_dir="./custom_vector_dbs"
)

# 修改文档分块策略
processor = DocumentProcessor()
processor.text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,  # 增大分块大小
    chunk_overlap=400  # 增大重叠区域
)
```

### 并行处理
对于大量仓库，可以考虑实现并行处理：

```python
import concurrent.futures

def parallel_build(repo_list, max_workers=4):
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(builder.build_single_repository, repo) 
                  for repo in repo_list]
        
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
```

---

*这个工具为您的代码仓库建立了强大的向量搜索能力，支持多种搜索策略和完整的Git历史上下文！*

**更新时间**: 2025-06-14
