#!/usr/bin/env python3
"""
Git提交信息后处理脚本
用于为repomix生成的markdown文档添加Git提交历史信息
"""

import re
import sys
from datetime import datetime
from pathlib import Path
import argparse
import git  # GitPython库


class GitCommitProcessor:
    def __init__(self, repo_path=".", md_file_path=None):
        """
        初始化处理器

        Args:
            repo_path: Git仓库路径
            md_file_path: markdown文件路径
        """
        self.repo_path = Path(repo_path)
        self.md_file_path = Path(md_file_path) if md_file_path else None
        self.commit_count = 5  # 默认显示5条提交历史

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
            # 获取文件最近的多个提交信息
            commits = list(repo.iter_commits(paths=file_path, max_count=count))

            commit_list = []
            for commit in commits:
                # 获取此次提交中对该文件的具体修改
                diffs = []
                if len(commit.parents) > 0:  # 如果不是初始提交
                    parent = commit.parents[0]
                    diffs = parent.diff(commit, paths=file_path)
                else:  # 初始提交
                    diffs = commit.diff(git.NULL_TREE, paths=file_path)

                # 提取文件变更内容
                diff_content = ""
                for diff_item in diffs:
                    if diff_item.a_path == file_path or diff_item.b_path == file_path:
                        if diff_item.diff == b"":  # Git 返回空 diff 表示二进制文件
                            diff_content = "[二进制文件]"
                        else:
                            diff_content = diff_item.diff

                commit_list.append(
                    {
                        "hash": commit.hexsha[:8],  # 短hash
                        "author": commit.author.name,
                        "email": commit.author.email,
                        "date": datetime.fromtimestamp(
                            commit.authored_date
                        ).isoformat(),
                        "message": commit.message.strip(),
                        "diff": diff_content,
                    }
                )

            return commit_list

        except Exception as e:
            print(f"获取文件 {file_path} 的提交历史时出错: {e}")
            return []

    def format_commit_info(self, commit_info):
        """
        格式化单个提交信息为markdown

        Args:
            commit_info: 提交信息字典

        Returns:
            str: 格式化后的markdown字符串
        """
        if not commit_info:
            return "\n**Git信息:** 无法获取提交历史\n"

        # 格式化日期
        try:
            date_obj = datetime.fromisoformat(commit_info["date"].replace(" +", "+"))
            formatted_date = date_obj.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            formatted_date = commit_info["date"]

        return f"""
**Git信息:**
- **最近提交:** `{commit_info["hash"]}`
- **提交者:** {commit_info["author"]} ({commit_info["email"]})
- **提交时间:** {formatted_date}
- **提交信息:** {commit_info["message"]}
"""

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
            # 格式化日期
            try:
                date_obj = datetime.fromisoformat(commit["date"].replace(" +", "+"))
                formatted_date = date_obj.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                formatted_date = commit["date"]

            result += f"""
#### 提交 {i + 1}
- **提交标识:** `{commit["hash"]}`
- **提交者:** {commit["author"]} ({commit["email"]})
- **提交时间:** {formatted_date}
- **提交信息:** {commit["message"]}

<details>
<summary>查看详细改动</summary>

```diff
{commit["diff"]}
```

</details>
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

        # 匹配文件路径模式 - 支持多种可能的格式
        # 1. 二级标题形式: ## File: path/to/file
        # 2. 列表项形式: * path/to/file
        # 3. 只有文件路径的形式

        # 尝试匹配二级标题形式
        file_header_pattern = r"## File:\s*(.+?)(?=\n)"
        file_headers = re.finditer(file_header_pattern, content)

        lines = content.split("\n")
        for match in file_headers:
            file_path = match.group(1).strip()
            line_start = content[: match.start()].count("\n")

            # 查找该文件部分的代码块
            code_blocks = []
            for i in range(line_start + 1, len(lines)):
                if i >= len(lines):
                    break

                # 如果遇到下一个文件标题，结束搜索
                if lines[i].startswith("## File:"):
                    break

                # 找到代码块开始
                if lines[i].startswith("```"):
                    code_start = i
                    # 寻找代码块结束
                    for j in range(code_start + 1, len(lines)):
                        if j >= len(lines):
                            break
                        if lines[j].startswith("```"):
                            code_end = j
                            code_blocks.append({"start": code_start, "end": code_end})
                            break

            if code_blocks:
                # 取最后一个代码块作为文件内容块
                last_code_block = code_blocks[-1]
                files.append(
                    {
                        "path": file_path,
                        "line_start": line_start,
                        "code_start": last_code_block["start"],
                        "code_end": last_code_block["end"],
                        "line_end": last_code_block["end"],
                    }
                )

        # 如果没有找到二级标题形式，尝试原来的列表项形式
        if not files:
            file_pattern = r"^\s*\*\s+(.+?)$"
            code_block_pattern = r"^```\s*$"

            i = 0
            while i < len(lines):
                line = lines[i]

                # 检查是否是文件路径行
                file_match = re.match(file_pattern, line)
                if file_match:
                    file_path = file_match.group(1).strip()

                    # 跳过不是实际文件路径的项（如 "文件路径" 这样的标题）
                    if file_path == "文件路径" or file_path == "...":
                        i += 1
                        continue

                    # 查找对应的代码块
                    j = i + 1
                    code_start = -1
                    code_end = -1

                    # 寻找代码块开始
                    while j < len(lines):
                        if re.match(code_block_pattern, lines[j]):
                            code_start = j
                            break
                        j += 1

                    # 寻找代码块结束
                    if code_start != -1:
                        j = code_start + 1
                        while j < len(lines):
                            if re.match(code_block_pattern, lines[j]):
                                code_end = j
                                break
                            j += 1

                    if code_start != -1 and code_end != -1:
                        files.append(
                            {
                                "path": file_path,
                                "line_start": i,
                                "code_start": code_start,
                                "code_end": code_end,
                                "line_end": code_end,
                            }
                        )
                        i = code_end + 1
                    else:
                        i += 1
                else:
                    i += 1

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

        # 如果没有找到文件部分，尝试使用正则表达式方法
        if not files:
            # 查找"Files"一级标题部分
            files_section_match = re.search(
                r"# Files.*?(?=\n# |$)", md_content, re.DOTALL
            )
            if files_section_match:
                files_section = files_section_match.group(0)
                # 在Files部分中寻找文件标题和代码块
                section_pattern = r"## File: (.*?)(?=\n## File:|$)"
                sections = re.findall(section_pattern, files_section, re.DOTALL)

                processed_md = md_content

                for section in sections:
                    file_path_line = section.strip().split("\n")[0].strip()
                    # 提取实际的文件路径
                    if ":" in file_path_line:
                        file_path = file_path_line.split(":", 1)[1].strip()
                    else:
                        file_path = file_path_line.strip()

                    # 确保代码块后插入git信息
                    code_blocks = re.findall(r"```.*?```", section, re.DOTALL)
                    if code_blocks:
                        last_code_block = code_blocks[-1]
                        print(f"处理文件: {file_path}")

                        # 获取指定数量的提交历史
                        commit_list = self.get_file_commit_history(
                            file_path, self.commit_count
                        )
                        git_history_text = self.format_commit_history(commit_list)

                        # 定位代码块在原文档中的位置
                        section_start = md_content.find(section)
                        block_in_section = section.find(last_code_block) + len(
                            last_code_block
                        )
                        insert_pos = section_start + block_in_section

                        if insert_pos > 0:
                            before = processed_md[:insert_pos]
                            after = processed_md[insert_pos:]
                            processed_md = before + "\n" + git_history_text + after

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


def main():
    parser = argparse.ArgumentParser(
        description="为repomix生成的markdown文档添加Git提交信息",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  %(prog)s repo.md                    # 处理repo.md文件，自动备份原文件
  %(prog)s repo.md -o output.md       # 输出到新文件
  %(prog)s repo.md -r /path/to/repo   # 指定Git仓库路径
  %(prog)s repo.md -c 3               # 指定每个文件显示3条提交历史（默认为5条）
        """,
    )

    parser.add_argument("input_file", help="输入的markdown文件路径")
    parser.add_argument("-o", "--output", help="输出文件路径（可选，默认覆盖原文件）")
    parser.add_argument(
        "-r", "--repo", default=".", help="Git仓库路径（默认为当前目录）"
    )
    parser.add_argument(
        "-c",
        "--commit-count",
        type=int,
        default=5,
        help="显示的提交历史数量（默认为5条）",
    )

    args = parser.parse_args()

    try:
        processor = GitCommitProcessor(repo_path=args.repo)
        # 设置提交历史数量
        processor.commit_count = args.commit_count
        processor.process_file(args.input_file, args.output)

    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
