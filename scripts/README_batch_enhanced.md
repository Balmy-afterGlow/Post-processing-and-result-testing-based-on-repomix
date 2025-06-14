# Token 计算增强版批量生成脚本

这个增强版的 `batch_generate_md.py` 脚本不仅可以批量生成 markdown 文件，还能计算 token 数量并生成详细的元数据报告。

## 新增功能

### 1. Token 计算
- 支持 GPT-4 和 GPT-3.5-turbo 模型的 token 计算
- 使用 tiktoken 库进行精确计算
- 提供简单估算作为备选方案

### 2. 元数据收集
每个生成的文件都会收集以下元数据：
- 文件基本信息（大小、行数、字符数等）
- 多种模型的 token 数量
- 创建和修改时间
- 压缩比率

### 3. 统计报告
- 每个仓库的详细元数据（`metadata.json`）
- 全局统计报告（`global_statistics.json`）
- 成功率和失败报告

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

```bash
python batch_generate_md.py
```

## 输出文件结构

```
../repomix_md/
├── repository-{name1}/
│   ├── repomix-output-{name1}.md
│   ├── repomix-output-{name1}-compress.md
│   └── metadata.json
├── repository-{name2}/
│   ├── repomix-output-{name2}.md
│   ├── repomix-output-{name2}-compress.md
│   └── metadata.json
└── global_statistics.json
```

## 元数据文件示例

### metadata.json
```json
{
  "repository": {
    "name": "owner/repo",
    "sha": "commit_hash",
    "generation_date": "2025-06-14T10:30:00"
  },
  "files": {
    "standard": {
      "file_name": "repomix-output-repo.md",
      "file_size_mb": 2.5,
      "token_counts": {
        "gpt-4": 12000,
        "gpt-3.5-turbo": 12100,
        "estimated": 11800
      },
      "character_count": 48000,
      "line_count": 1200
    },
    "compressed": {
      "file_name": "repomix-output-repo-compress.md",
      "file_size_mb": 1.8,
      "token_counts": {
        "gpt-4": 8500,
        "gpt-3.5-turbo": 8600,
        "estimated": 8200
      }
    }
  },
  "summary": {
    "compression_ratio": {
      "size": 0.72,
      "tokens": 0.71
    }
  }
}
```

### global_statistics.json
```json
{
  "generation_summary": {
    "total_repositories": 50,
    "successful": 48,
    "failed": 2,
    "success_rate": 96.0
  },
  "global_statistics": {
    "total_files_generated": 96,
    "total_size_mb": 125.6,
    "total_tokens_gpt4": 580000,
    "average_compression_ratio": 0.68
  },
  "failed_repositories": [
    {
      "repo": "owner/problematic-repo",
      "error": "Repository not found"
    }
  ]
}
```

## Token 计算说明

1. **GPT-4/GPT-3.5-turbo**: 使用 tiktoken 库进行精确计算
2. **估算方法**: 约4个字符=1个token（作为备用）
3. **压缩比率**: 压缩版本相对于标准版本的 token 比率

## 注意事项

1. 确保安装了 `tiktoken` 库
2. 网络连接良好（需要访问远程仓库）
3. 有足够的磁盘空间存储生成的文件
4. Token 计算可能需要一些时间，特别是对于大文件

## 错误处理

- 脚本会自动跳过无法访问的仓库
- 生成详细的错误报告
- 继续处理其他仓库，不会因单个失败而停止
