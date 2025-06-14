#!/usr/bin/env python3
"""
Git提交信息后处理脚本
用于为repomix生成的markdown文档添加Git提交历史信息
"""

import re
import sys
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
import argparse
import git  # GitPython库


class GitCommitProcessor:
    def __init__(
        self, repo_path=".", md_file_path=None, remote_repo=None, target_commit=None
    ):
        """
        初始化处理器

        Args:
            repo_path: Git仓库路径（本地）
            md_file_path: markdown文件路径
            remote_repo: 远程仓库（格式：owner/repo 或完整URL）
            target_commit: 目标提交SHA，用于检出到特定状态获取文件历史
        """
        self.md_file_path = Path(md_file_path) if md_file_path else None
        self.commit_count = 5  # 默认显示5条提交历史
        self.remote_repo = remote_repo
        self.target_commit = target_commit
        self.temp_dir = None
        self.repo_path = None

        # 处理远程仓库
        if remote_repo:
            self._setup_remote_repo()
        else:
            self.repo_path = Path(repo_path)

    def _setup_remote_repo(self):
        """设置远程仓库，克隆到临时目录"""
        try:
            # 创建临时目录
            self.temp_dir = tempfile.mkdtemp(prefix="git_commit_processor_")
            print(f"创建临时目录: {self.temp_dir}")

            # 构建远程仓库URL
            if self.remote_repo.startswith("http"):
                repo_url = self.remote_repo
            elif "/" in self.remote_repo and not self.remote_repo.startswith("/"):
                # 格式如 owner/repo
                repo_url = f"https://github.com/{self.remote_repo}.git"
            else:
                raise ValueError(f"无效的远程仓库格式: {self.remote_repo}")

            print(f"正在克隆远程仓库: {repo_url}")
            # 克隆仓库到临时目录
            repo = git.Repo.clone_from(repo_url, self.temp_dir)
            self.repo_path = Path(self.temp_dir)
            print(f"仓库克隆完成: {self.temp_dir}")

            # 如果指定了目标提交，检出到该提交
            if self.target_commit:
                print(f"正在检出到指定提交: {self.target_commit}")
                try:
                    repo.git.checkout(self.target_commit)
                    current_commit = repo.head.commit
                    print(f"成功检出到提交: {current_commit.hexsha[:8]}")
                    print(f"提交信息: {current_commit.message.strip()}")
                    print(
                        f"提交时间: {datetime.fromtimestamp(current_commit.authored_date).strftime('%Y-%m-%d %H:%M:%S')}"
                    )
                except Exception as e:
                    print(f"警告: 无法检出到指定提交 {self.target_commit}: {e}")
                    print("将使用默认分支进行处理")

        except Exception as e:
            if self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
            raise Exception(f"克隆远程仓库失败: {e}")

    def cleanup(self):
        """清理临时目录"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            print(f"清理临时目录: {self.temp_dir}")
            shutil.rmtree(self.temp_dir)
            self.temp_dir = None

    def __enter__(self):
        """支持上下文管理器"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出时自动清理"""
        self.cleanup()

    def get_file_commit_history(self, file_path, count=5):
        """
        获取文件的多个提交历史

        Args:
            file_path: 文件相对路径
            count: 获取的提交记录数量

        Returns:
            list: 包含多个提交信息的列表，如果获取失败返回空列表
        """
        try:
            repo = git.Repo(self.repo_path)

            # 如果指定了目标提交，从该提交开始获取历史
            if self.target_commit:
                try:
                    # 从指定提交开始获取文件历史
                    commits = list(
                        repo.iter_commits(
                            self.target_commit, paths=file_path, max_count=count
                        )
                    )
                    if not commits:
                        print(
                            f"警告: 在提交 {self.target_commit} 中未找到文件 {file_path}，尝试获取全局历史"
                        )
                        commits = list(
                            repo.iter_commits(paths=file_path, max_count=count)
                        )
                except Exception as e:
                    print(f"从指定提交获取历史失败: {e}，使用默认方法")
                    commits = list(repo.iter_commits(paths=file_path, max_count=count))
            else:
                # 获取文件最近的多个提交信息
                commits = list(repo.iter_commits(paths=file_path, max_count=count))

            commit_list = []
            for commit in commits:
                # 获取此次提交中对该文件的具体修改
                diffs = []
                if len(commit.parents) > 0:  # 如果不是初始提交
                    parent = commit.parents[0]
                    diffs = parent.diff(commit, paths=file_path, create_patch=True)
                else:  # 初始提交
                    diffs = commit.diff(
                        git.NULL_TREE, paths=file_path, create_patch=True
                    )

                # 提取文件变更内容
                diff_content = ""
                for diff_item in diffs:
                    if diff_item.a_path == file_path or diff_item.b_path == file_path:
                        try:
                            diff_content = diff_item.diff.decode("utf-8")
                        except UnicodeDecodeError:
                            diff_content = "[二进制文件或编码不支持]"

                commit_list.append(
                    {
                        "hash": commit.hexsha[:8],  # 短hash
                        "author": commit.author.name,
                        "email": commit.author.email,
                        "date": datetime.fromtimestamp(commit.authored_date).strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        "message": commit.message.strip(),
                        "diff": diff_content,
                    }
                )

            return commit_list

        except Exception as e:
            print(f"获取文件 {file_path} 的提交历史时出错: {e}")
            return []

    def format_commit_history(self, commit_list):
        """
        格式化多个提交历史为markdown

        Args:
            commit_list: 提交历史列表

        Returns:
            str: 格式化后的markdown字符串
        """
        if not commit_list:
            return "\n**Git提交历史:** 无法获取提交历史\n"

        result = "\n### Git提交历史\n"

        for i, commit in enumerate(commit_list):
            result += f"""
#### 提交 {i + 1}
- **提交标识:** `{commit["hash"]}`
- **提交者:** {commit["author"]} ({commit["email"]})
- **提交时间:** {commit["date"]}
- **提交信息:** {commit["message"]}


详细改动如下：

```diff
{commit["diff"]}
```

"""
        return result

    def extract_file_sections(self, content):
        """
        从markdown内容中提取文件部分

        Args:
            content: markdown文件内容

        Returns:
            list: 包含文件信息的列表
        """
        files = []

        # 匹配文件路径模式 - 支持二级标题形式: ## File: path/to/file

        # 尝试匹配二级标题形式
        file_header_pattern = r"## File:\s*(.+?)(?=\n)"
        file_headers = re.finditer(file_header_pattern, content)

        lines = content.split("\n")
        for match in file_headers:
            file_path = match.group(1).strip()
            line_start = content[: match.start()].count("\n")
            line_end = -1

            # 查找该文件部分的代码块
            code_blocks = []
            i = line_start + 1
            while i < len(lines):
                # 如果遇到下一个文件标题，结束搜索
                if lines[i].startswith("## File:"):
                    line_end = i - 1
                    break

                # 找到代码块开始
                if lines[i].startswith("```"):
                    code_start = i
                    # 寻找代码块结束
                    for j in range(code_start + 1, len(lines)):
                        if lines[j].startswith("```"):
                            code_end = j
                            code_blocks.append({"start": code_start, "end": code_end})
                            i = code_end + 1
                            break
                    continue
                i += 1

            if code_blocks:
                # 取最后一个代码块作为文件内容块
                last_code_block = code_blocks[-1]
                files.append(
                    {
                        "path": file_path,
                        "line_start": line_start,
                        "line_end": line_end if line_end != -1 else len(lines),
                        "code_start": last_code_block["start"],
                        "code_end": last_code_block["end"],
                    }
                )

        return files

    def process_markdown(self, md_content):
        """
        处理markdown内容，添加Git提交信息

        Args:
            md_content: 原始markdown内容

        Returns:
            str: 处理后的markdown内容
        """
        # 首先使用提取的文件部分
        files = self.extract_file_sections(md_content)
        for file_info in files:
            print(f"处理文件: {file_info['path']}")
        lines = md_content.split("\n")

        # 按行号从后往前处理，避免行号变化影响
        for file_info in reversed(files):
            file_path = file_info["path"]
            print(f"处理文件: {file_path}")

            # 获取指定数量的提交历史
            commit_list = self.get_file_commit_history(file_path, self.commit_count)
            git_history_text = self.format_commit_history(commit_list)

            # 在代码块结束后插入Git信息
            insert_position = file_info["code_end"] + 1
            git_info_lines = git_history_text.split("\n")

            # 在指定位置插入Git历史信息
            for j, git_line in enumerate(reversed(git_info_lines)):
                lines.insert(insert_position, git_line)

        # 重新组合处理后的内容
        processed_md = "\n".join(lines)

        return processed_md

    def process_file(self, input_file, output_file=None):
        """
        处理markdown文件

        Args:
            input_file: 输入文件路径
            output_file: 输出文件路径，如果为None则覆盖原文件
        """
        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"输入文件不存在: {input_file}")

        # 检查仓库是否有效
        if not self.repo_path or not self.repo_path.exists():
            raise FileNotFoundError(f"Git仓库路径不存在: {self.repo_path}")

        try:
            # 验证这是一个有效的Git仓库
            repo = git.Repo(self.repo_path)
            print(f"使用Git仓库: {self.repo_path}")

            try:
                current_commit = repo.head.commit
                print(
                    f"当前提交: {current_commit.hexsha[:8]} - {current_commit.message.strip()[:50]}"
                )
                if self.target_commit:
                    if current_commit.hexsha.startswith(self.target_commit):
                        print(f"✅ 已成功检出到目标提交: {self.target_commit}")
                    else:
                        print("⚠️  当前提交与目标提交不匹配")

                try:
                    print(f"当前分支: {repo.active_branch.name}")
                except TypeError:
                    print("当前状态: detached HEAD (已检出到特定提交)")
            except Exception as e:
                print(f"获取当前状态信息失败: {e}")

        except git.exc.InvalidGitRepositoryError:
            raise ValueError(f"指定路径不是有效的Git仓库: {self.repo_path}")

        # 读取原文件
        with open(input_path, "r", encoding="utf-8") as f:
            content = f.read()

        print(f"开始处理文件: {input_path}")

        # 处理内容
        processed_content = self.process_markdown(content)

        # 确定输出文件路径
        if output_file:
            output_path = Path(output_file)
        else:
            # 创建备份
            backup_path = input_path.with_suffix(input_path.suffix + ".backup")
            input_path.rename(backup_path)
            output_path = input_path
            print(f"原文件已备份为: {backup_path}")

        # 写入处理后的内容
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(processed_content)

        print(f"处理完成，输出文件: {output_path}")

        if self.remote_repo:
            print("远程仓库信息已添加完成，临时目录将在脚本结束时清理")


def main():
    parser = argparse.ArgumentParser(
        description="为repomix生成的markdown文档添加Git提交信息",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  %(prog)s repo.md                    # 处理repo.md文件，自动备份原文件
  %(prog)s repo.md -o output.md       # 输出到新文件
  %(prog)s repo.md -r /path/to/repo   # 指定本地Git仓库路径
  %(prog)s repo.md -r owner/repo      # 指定GitHub远程仓库（格式：owner/repo）
  %(prog)s repo.md -r https://github.com/owner/repo.git  # 指定完整的远程仓库URL
  %(prog)s repo.md -c 3               # 指定每个文件显示3条提交历史（默认为5条）
  %(prog)s repo.md -r owner/repo --target-commit abc123  # 从指定提交开始获取历史
        """,
    )

    parser.add_argument("input_file", help="输入的markdown文件路径")
    parser.add_argument("-o", "--output", help="输出文件路径（可选，默认覆盖原文件）")
    parser.add_argument(
        "-r",
        "--repo",
        help="Git仓库路径。可以是本地路径、GitHub格式(owner/repo)或完整URL",
    )
    parser.add_argument(
        "-c",
        "--commit-count",
        type=int,
        default=5,
        help="显示的提交历史数量（默认为5条）",
    )
    parser.add_argument(
        "--target-commit",
        help="目标提交SHA（完整或前缀），从该提交状态开始获取文件历史",
    )

    args = parser.parse_args()

    try:
        # 判断是否为远程仓库
        is_remote = False
        if args.repo:
            # 检查是否为远程仓库格式
            if args.repo.startswith("http") or (
                "/" in args.repo
                and not args.repo.startswith("/")
                and not os.path.exists(args.repo)
            ):
                is_remote = True

        if is_remote:
            # 使用上下文管理器处理远程仓库
            with GitCommitProcessor(
                remote_repo=args.repo, target_commit=args.target_commit
            ) as processor:
                processor.commit_count = args.commit_count
                processor.process_file(args.input_file, args.output)
        else:
            # 处理本地仓库
            repo_path = args.repo if args.repo else "."
            processor = GitCommitProcessor(
                repo_path=repo_path, target_commit=args.target_commit
            )
            processor.commit_count = args.commit_count
            processor.process_file(args.input_file, args.output)

    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
