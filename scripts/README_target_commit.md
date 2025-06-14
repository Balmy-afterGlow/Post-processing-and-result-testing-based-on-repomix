# Git 提交历史脚本 - 支持指定提交状态

`add_commit_info.py` 脚本现在支持根据指定的提交hash来获取文件的提交历史。这个功能对于分析特定时间点的代码状态非常有用。

## 新增功能

### 指定目标提交参数

新增了 `--target-commit` 参数，允许您指定一个提交hash，脚本会：

1. **检出到指定提交**: 将仓库状态切换到指定的提交
2. **从该提交开始获取历史**: 获取文件在该提交状态下的历史记录
3. **标记目标提交**: 在输出中明确标记哪个是目标提交

## 使用方法

### 基本语法
```bash
python add_commit_info.py <markdown_file> [选项] --target-commit <commit_hash>
```

### 使用示例

#### 1. 远程仓库 + 指定提交
```bash
python add_commit_info.py repo.md -r microsoft/vscode --target-commit abc123456 -o output.md
```

#### 2. 本地仓库 + 指定提交
```bash
python add_commit_info.py repo.md -r /path/to/local/repo --target-commit abc123456
```

#### 3. 完整URL + 指定提交
```bash
python add_commit_info.py repo.md -r https://github.com/owner/repo.git --target-commit abc123456
```

#### 4. 配合其他参数
```bash
python add_commit_info.py repo.md -r owner/repo --target-commit abc123456 -c 3 -o enhanced.md
```

## 参数说明

| 参数 | 描述 | 示例 |
|------|------|------|
| `--target-commit` | 目标提交SHA（完整或前缀） | `abc123456` 或 `abc123` |
| `-r, --repo` | 仓库路径或URL | `owner/repo` |
| `-c, --commit-count` | 显示的提交历史数量 | `3` |
| `-o, --output` | 输出文件路径 | `output.md` |

## 工作流程

1. **克隆仓库** (如果是远程仓库)
2. **检出到目标提交**
   ```
   正在检出到指定提交: abc123456
   成功检出到提交: abc12345
   提交信息: Fix critical bug in authentication
   提交时间: 2025-06-14 10:30:00
   ```
3. **验证检出状态**
   ```
   ✅ 已成功检出到目标提交: abc123456
   当前状态: detached HEAD (已检出到特定提交)
   ```
4. **获取文件历史** - 从目标提交开始向前获取历史
5. **生成增强的markdown** - 包含提交历史信息

## 输出特点

### 状态信息显示
生成的markdown文件会包含以下信息：

```markdown
### Git提交历史

> **注意**: 以下提交历史是从指定提交 `abc123456` 开始获取的

#### 提交 1 🎯 **[目标提交]**
- **提交标识:** `abc12345`
- **提交者:** John Doe (john@example.com)
- **提交时间:** 2025-06-14 10:30:00
- **提交信息:** Fix critical bug in authentication

详细改动如下：
```diff
...具体的代码差异...
```

#### 提交 2
- **提交标识:** `def67890`
...
```

### 特殊标记
- **🎯 [目标提交]**: 标记指定的目标提交
- **注意提示**: 说明历史是从指定提交开始获取的
- **状态验证**: 显示当前检出状态

## 错误处理

### 提交不存在
```bash
警告: 无法检出到指定提交 invalidhash: reference 'invalidhash' not found
将使用默认分支进行处理
```

### 文件在指定提交中不存在
```bash
警告: 在提交 abc123456 中未找到文件 src/newfile.js，尝试获取全局历史
```

### 网络问题
脚本会自动重试和提供详细的错误信息。

## 使用场景

1. **Bug分析**: 从引入bug的提交开始分析代码演进
2. **功能追踪**: 从添加特定功能的提交开始查看后续修改
3. **版本比较**: 比较不同版本时期的代码状态
4. **历史研究**: 深入了解代码在特定时间点的状态

## 注意事项

1. **提交hash**: 可以使用完整的40位hash或前缀（至少4位）
2. **detached HEAD**: 检出到特定提交会处于detached HEAD状态，这是正常的
3. **历史方向**: 从指定提交开始**向前**获取历史（包含该提交及之前的提交）
4. **文件存在性**: 如果文件在指定提交中不存在，会回退到全局历史

这个增强功能让您可以精确地分析代码在任何特定时间点的状态和演进历史！
