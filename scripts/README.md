# 🚀 Repository Documentation Generator

代码仓库文档化工具脚本，能够自动批量生成包含Git提交历史的markdown文档，为AI模型提供丰富的上下文信息。

## 🌟 主要功能

### 核心能力
- ✅ **批量处理多个GitHub仓库** - 自动化处理大量仓库
- ✅ **生成三种版本文档** - 标准版、压缩版、Git增强版
- ✅ **Git提交历史集成** - 为每个文件添加详细的提交历史
- ✅ **智能元数据统计** - Token计数、文件大小、压缩比分析
- ✅ **指定提交状态支持** - 从特定commit开始获取历史
- ✅ **远程仓库直接处理** - 无需本地克隆，支持GitHub URL
- ✅ **实时进度监控** - 详细的输出信息和错误处理

## ⚡ 快速开始

### 1. 环境准备
```bash
# 通过conda来管理虚拟环境
conda create -n repomind python=3.12

# 安装Python依赖
cd path/to/project_root && pip install -r requirements.txt

# 安装repomix工具
npm install -g repomix
```

### 2. 批量生成文档（推荐方式）—— 内部使用了`add_commit_info.py`
```bash
# 自动处理所有仓库，生成三种版本
python batch_generate_md.py

# 输出目录结构
# ../repomix_md/
# ├── repository-vscode/
# │   ├── repomix-output-vscode.md           # 📄 标准版
# │   ├── repomix-output-vscode-compress.md  # 🗜️ 压缩版
# │   ├── repomix-output-vscode-with-git.md  # 🔄 Git增强版
# │   └── metadata.json                      # 📊 详细统计
# └── ...
```

### 3. 单独添加Git历史
```bash
# 为现有文档添加Git历史
python add_commit_info.py document.md -r microsoft/vscode -o enhanced.md

# 从特定提交开始获取历史
python add_commit_info.py document.md -r owner/repo --target-commit abc123456
```

## 🛠️ 核心脚本

| 脚本文件 | 主要功能 | 使用场景 |
|---------|---------|---------|
| `batch_generate_md.py` | 批量生成多版本文档 | 🏭 大规模文档生成 |
| `add_commit_info.py` | 单文档Git历史增强 | 🎯 精确控制单个文档 |

## 🔧 安装配置

### 系统要求
- **操作系统**: Linux
- **Python**: 3.9+ (推荐 3.12)
- **Node.js**: 16+ (运行repomix)
- **Git**: 2.20+ (系统命令行工具)
- **网络**: 稳定连接(处理远程仓库)

### 依赖安装

#### Python依赖
```bash
# 通过conda来管理虚拟环境（默认是repomind）
conda create -n repomind python=3.12

# 启动虚拟环境
conda activate repomind

# 安装Python依赖
cd path/to/project_root && pip install -r requirements

# 验证安装
conda list
```

#### Node.js依赖
```bash
# 全局安装repomix
npm install -g repomix

# 验证安装
repomix --help
```

### 配置文件

#### 仓库列表配置
`batch_generate_md.py`脚本从 `../evalulate/datasets/issues.json` 读取仓库列表，格式如下：
```json
[
  {
    "repo": "microsoft/vscode",
    "base_sha": "abc123456789...",
    "other_fields": "..."
  },
  {
    "repo": "facebook/react",
    "base_sha": "def987654321...",
    "other_fields": "..."
  }
]
```

#### 自定义配置选项
```python
# 在`batch_generate_md.py`脚本中可以修改的配置
ignore_patterns = "*.md,*.MD,*.ipynb,docs/*,test/*,tests/*,examples/*"
env_name = "your_conda_env"  # 如果使用conda
commit_count = 5  # Git历史显示数量
```

## 📖 详细使用指南

### 1. 批量文档生成

#### 基本用法
```bash
# 处理所有配置的仓库
python batch_generate_md.py
```

#### 输出说明
每个仓库会生成以下文件：
```
repository-{name}/
├── repomix-output-{name}.md              # 标准版本
├── repomix-output-{name}-compress.md     # 压缩版本 (约30-50%体积减少)
├── repomix-output-{name}-with-git.md     # Git增强版本 (包含提交历史)
└── metadata.json                         # 详细统计信息
```

#### metadata.json 详解
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

### 2. Git历史增强

#### 基本语法
```bash
python add_commit_info.py <input_file> [options]
```

#### 参数详解
| 参数 | 短参数 | 描述 | 示例 | 必需 |
|------|-------|------|------|------|
| `input_file` | - | 输入的markdown文件 | `repo.md` | ✅ |
| `--repo` | `-r` | 仓库路径或URL | `owner/repo` | ⚪ |
| `--target-commit` | - | 目标提交SHA | `abc123456` | ⚪ |
| `--commit-count` | `-c` | 显示的提交数量 | `3` | ⚪ |
| `--output` | `-o` | 输出文件路径 | `output.md` | ⚪ |

#### 仓库格式支持
```bash
# GitHub短格式
python add_commit_info.py repo.md -r microsoft/vscode

# 完整GitHub URL
python add_commit_info.py repo.md -r https://github.com/microsoft/vscode.git

# SSH格式
python add_commit_info.py repo.md -r git@github.com:microsoft/vscode.git

# 本地仓库路径
python add_commit_info.py repo.md -r /path/to/local/repo

# 使用完整commit hash
python add_commit_info.py repo.md -r owner/repo --target-commit a1b2c3d4e5f6g7h8

# 使用短hash（推荐至少7位）
python add_commit_info.py repo.md -r owner/repo --target-commit a1b2c3d
```

#### 复杂用法示例
```bash
# 组合多个参数
python add_commit_info.py repo.md \
  --repo microsoft/vscode \
  --target-commit abc123456 \
  --commit-count 3 \
  --output vscode-enhanced.md
```

#### 输出说明
````markdown
# Files

## File: Path/to/File
...

### Git提交历史

#### 提交 1
- **提交标识:** `abc123456`
- **提交者:** John Doe (john@example.com)
- **提交时间:** 2025-06-14 10:30:00
- **提交信息:** Fix critical authentication bug

详细改动如下：
```diff
+ 修复后的代码
- 原有的错误代码
```

#### 提交 2
- **提交标识:** `def789012`
- **提交者:** Jane Smith (jane@example.com)
- **提交时间:** 2025-06-13 15:20:00
- **提交信息:** Add user authentication feature
...
````

**最后更新**: 2025-06-14
**版本**: 1.0.0
