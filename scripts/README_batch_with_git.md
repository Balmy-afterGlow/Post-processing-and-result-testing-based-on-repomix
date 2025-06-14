# 增强版批量生成脚本 - 支持 Git 提交历史

这个增强版的 `batch_generate_md_enhanced.py` 脚本现在集成了 `add_commit_info.py` 的功能，能够自动生成包含 Git 提交历史的 markdown 文件。

## 新增功能

### 自动生成三种版本的文件
1. **标准版本** (`repomix-output-{name}.md`) - 原始的 repomix 输出
2. **压缩版本** (`repomix-output-{name}-compress.md`) - 压缩后的版本
3. **Git增强版本** (`repomix-output-{name}-with-git.md`) - 包含 Git 提交历史的版本

### 集成的 Git 功能
- 自动调用 `add_commit_info.py` 脚本
- 为每个文件添加最近 3 条提交历史
- 包含提交者信息、时间、消息和详细的 diff 内容
- 自动处理远程仓库（无需本地克隆）

## 使用方法

```bash
python batch_generate_md_enhanced.py
```

## 输出文件结构

```
../repomix_md/
├── repository-{name1}/
│   ├── repomix-output-{name1}.md              # 标准版本
│   ├── repomix-output-{name1}-compress.md     # 压缩版本
│   ├── repomix-output-{name1}-with-git.md     # Git增强版本
│   └── metadata.json                          # 元数据（包含三个文件的信息）
├── repository-{name2}/
│   ├── repomix-output-{name2}.md
│   ├── repomix-output-{name2}-compress.md
│   ├── repomix-output-{name2}-with-git.md
│   └── metadata.json
└── global_statistics.json                     # 全局统计（未实现，但结构已准备）
```

## 元数据示例

现在的 `metadata.json` 包含三个文件的信息：

```json
{
  "repository": {
    "name": "owner/repo",
    "sha": "abc123",
    "generation_date": "2025-06-14T15:30:00"
  },
  "files": {
    "standard": {
      "file_name": "repomix-output-repo.md",
      "file_size_mb": 2.5,
      "token_counts": {
        "gpt-4": 12000,
        "estimated": 11800
      }
    },
    "compressed": {
      "file_name": "repomix-output-repo-compress.md",
      "file_size_mb": 1.8,
      "token_counts": {
        "gpt-4": 8500,
        "estimated": 8200
      }
    },
    "git_enhanced": {
      "file_name": "repomix-output-repo-with-git.md",
      "file_size_mb": 4.2,
      "token_counts": {
        "gpt-4": 18000,
        "estimated": 16800
      }
    }
  },
  "summary": {
    "total_files": 3,
    "total_size_bytes": 8765432,
    "total_tokens_gpt4": 38500,
    "compression_ratio": {
      "size": 0.72,
      "tokens": 0.71
    }
  }
}
```

## 工作流程

1. **生成标准版本**: 使用 repomix 生成标准 markdown
2. **生成压缩版本**: 使用 repomix --compress 生成压缩版本
3. **生成Git增强版本**: 
   - 调用 `add_commit_info.py` 脚本
   - 传递远程仓库信息 (`-r {repo}`)
   - 设置显示 3 条提交历史 (`-c 3`)
4. **计算元数据**: 为所有生成的文件计算 token 数量和其他统计信息
5. **保存元数据**: 将完整的元数据保存为 JSON 文件

## Git 增强版本特点

- **自动化**: 无需手动运行 `add_commit_info.py`
- **远程支持**: 直接处理 GitHub 仓库，无需本地克隆
- **适度历史**: 只显示 3 条提交历史，平衡信息量和文件大小
- **错误处理**: 如果 Git 处理失败，仍会生成标准和压缩版本

## 依赖要求

```bash
pip install tiktoken GitPython
```

## 注意事项

1. **网络连接**: 需要良好的网络连接访问远程仓库
2. **处理时间**: Git 增强版本需要额外的处理时间
3. **文件大小**: Git 增强版本通常比标准版本大 50-100%
4. **错误恢复**: 即使 Git 处理失败，其他版本仍会正常生成

## 与原版本的区别

- ✅ 新增：自动生成 Git 增强版本
- ✅ 增强：元数据包含三个文件的信息
- ✅ 改进：更详细的进度显示和错误处理
- ✅ 保持：原有的 token 计算和压缩功能完全兼容

这个增强版本特别适合需要了解代码演进历史的场景，为 AI 模型提供更丰富的上下文信息！
