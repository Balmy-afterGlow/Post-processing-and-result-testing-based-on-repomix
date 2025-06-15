#!/usr/bin/env python3
"""
Git提交信息后处理脚本 - 优化版
用于为repomix生成的markdown文档添加Git提交历史信息
针对大型文档进行了性能优化
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
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from typing import List, Dict, Optional
import time


class OptimizedGitCommitProcessor:
    def __init__(
        self,
        repo_path=".",
        md_file_path=None,
        remote_repo=None,
        target_commit=None,
        max_workers=4,  # 并发处理数
        max_file_size_mb=50,  # 最大处理文件大小
        skip_patterns=None,  # 跳过的文件模式
        commit_count=3,  # 减少默认提交数量
    ):
        """
        初始化优化处理器

        Args:
            repo_path: Git仓库路径（本地）
            md_file_path: markdown文件路径
            remote_repo: 远程仓库（格式：owner/repo 或完整URL）
            target_commit: 目标提交SHA
            max_workers: 最大并发工作线程数
            max_file_size_mb: 跳过超过此大小的文件（MB）
            skip_patterns: 跳过的文件模式列表
            commit_count: 每个文件的提交历史数量
        """
        self.md_file_path = Path(md_file_path) if md_file_path else None
        self.commit_count = commit_count
        self.remote_repo = remote_repo
        self.target_commit = target_commit
        self.temp_dir = None
        self.repo_path = None
        self.max_workers = max_workers
        self.max_file_size_mb = max_file_size_mb

        # 默认跳过的文件模式
        self.skip_patterns = skip_patterns or [
            r"\.git/",
            r"node_modules/",
            r"\.venv/",
            r"__pycache__/",
            r"\.pyc$",
            r"\.jpg$",
            r"\.png$",
            r"\.gif$",
            r"\.ico$",
            r"\.pdf$",
            r"\.zip$",
            r"\.tar\.gz$",
            r"package-lock\.json$",
            r"yarn\.lock$",
            r"Pipfile\.lock$",
            r"\.min\.js$",
            r"\.min\.css$",
        ]

        # 缓存机制
        self._commit_cache = {}
        self._cache_lock = threading.Lock()

        # 统计信息
        self.stats = {
            "total_files": 0,
            "processed_files": 0,
            "skipped_files": 0,
            "cached_hits": 0,
            "processing_time": 0,
        }

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
            print(f"📁 创建临时目录: {self.temp_dir}")

            # 构建远程仓库URL
            if self.remote_repo.startswith("http"):
                repo_url = self.remote_repo
            elif "/" in self.remote_repo and not self.remote_repo.startswith("/"):
                repo_url = f"https://github.com/{self.remote_repo}.git"
            else:
                raise ValueError(f"无效的远程仓库格式: {self.remote_repo}")

            print(f"🔄 正在克隆远程仓库: {repo_url}")
            print("💡 提示: 大型仓库克隆可能需要一些时间...")

            # 使用浅克隆减少下载时间
            repo = git.Repo.clone_from(
                repo_url, self.temp_dir, depth=100
            )  # 只克隆最近100次提交
            self.repo_path = Path(self.temp_dir)
            print("✅ 仓库克隆完成")

            # 如果指定了目标提交，检出到该提交
            if self.target_commit:
                print(f"🎯 正在检出到指定提交: {self.target_commit}")
                try:
                    repo.git.checkout(self.target_commit)
                    current_commit = repo.head.commit
                    print(f"✅ 成功检出到提交: {current_commit.hexsha[:8]}")
                except Exception as e:
                    print(f"⚠️  警告: 无法检出到指定提交 {self.target_commit}: {e}")

        except Exception as e:
            if self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
            raise Exception(f"克隆远程仓库失败: {e}")

    def should_skip_file(self, file_path: str) -> bool:
        """检查是否应该跳过该文件"""
        # 检查文件模式
        for pattern in self.skip_patterns:
            if re.search(pattern, file_path):
                return True

        # 检查文件路径是否存在
        full_path = self.repo_path / file_path
        if full_path.exists() and full_path.is_file():
            # 检查文件大小
            file_size_mb = full_path.stat().st_size / (1024 * 1024)
            if file_size_mb > self.max_file_size_mb:
                print(f"⏭️  跳过大文件 ({file_size_mb:.1f}MB): {file_path}")
                return True

        return False

    def get_file_commit_history_cached(
        self, file_path: str, count: int = 3
    ) -> List[Dict]:
        """
        获取文件提交历史（带缓存）
        """
        cache_key = f"{file_path}:{count}"

        # 检查缓存
        with self._cache_lock:
            if cache_key in self._commit_cache:
                self.stats["cached_hits"] += 1
                return self._commit_cache[cache_key]

        # 获取提交历史
        commit_list = self._get_file_commit_history_direct(file_path, count)

        # 存入缓存
        with self._cache_lock:
            self._commit_cache[cache_key] = commit_list

        return commit_list

    def _get_file_commit_history_direct(self, file_path: str, count: int) -> List[Dict]:
        """直接获取文件提交历史"""
        try:
            repo = git.Repo(self.repo_path)

            # 优化：使用更快的git命令
            try:
                # 使用git log命令直接获取，比GitPython的iter_commits更快
                cmd = [
                    "log",
                    f"--max-count={count}",
                    "--pretty=format:%H|%an|%ae|%at|%s",
                    "--follow",  # 跟踪文件重命名
                    "--",
                    file_path,
                ]

                log_output = repo.git.execute(cmd)
                if not log_output.strip():
                    return []

                commit_list = []
                for line in log_output.strip().split("\n"):
                    if not line.strip():
                        continue

                    parts = line.split("|", 4)
                    if len(parts) != 5:
                        continue

                    commit_hash, author, email, timestamp, message = parts

                    # 只获取简化的diff（不获取完整patch）
                    try:
                        diff_output = repo.git.execute(
                            [
                                "show",
                                "--name-status",
                                "--pretty=format:",
                                commit_hash,
                                "--",
                                file_path,
                            ]
                        )
                        diff_content = (
                            diff_output.strip()
                            if diff_output.strip()
                            else "[无变更详情]"
                        )
                    except Exception:
                        diff_content = "[无法获取变更详情]"

                    commit_list.append(
                        {
                            "hash": commit_hash[:8],
                            "author": author,
                            "email": email,
                            "date": datetime.fromtimestamp(int(timestamp)).strftime(
                                "%Y-%m-%d %H:%M:%S"
                            ),
                            "message": message.strip(),
                            "diff": diff_content,
                        }
                    )

                return commit_list

            except Exception as e:
                print(f"⚠️  获取 {file_path} 提交历史失败: {e}")
                return []

        except Exception as e:
            print(f"❌ 处理 {file_path} 时出错: {e}")
            return []

    def format_commit_history_compact(self, commit_list: List[Dict]) -> str:
        """
        格式化提交历史为紧凑的markdown格式
        """
        if not commit_list:
            return "\n**Git提交历史:** 无法获取提交历史\n"

        result = "\n### Git提交历史\n"

        for i, commit in enumerate(commit_list):
            # 使用更紧凑的格式
            result += f"""
#### 提交 {i + 1}
- **提交标识:** `{commit["hash"]}`
- **提交者:** {commit["author"]}
- **提交时间:** {commit["date"]}
- **提交信息:** {commit["message"]}

```
{commit["diff"]}
```

"""
        return result

    def process_file_batch(self, file_infos: List[Dict]) -> List[Dict]:
        """
        批量处理文件信息
        """
        results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交任务
            future_to_file = {}
            for file_info in file_infos:
                if self.should_skip_file(file_info["path"]):
                    self.stats["skipped_files"] += 1
                    continue

                future = executor.submit(self._process_single_file_info, file_info)
                future_to_file[future] = file_info

            # 收集结果
            for future in as_completed(future_to_file):
                file_info = future_to_file[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        self.stats["processed_files"] += 1
                except Exception as e:
                    print(f"❌ 处理文件 {file_info['path']} 失败: {e}")

        return results

    def _process_single_file_info(self, file_info: Dict) -> Optional[Dict]:
        """
        处理单个文件信息
        """
        file_path = file_info["path"]
        print(f"🔄 处理文件: {file_path}")

        # 获取提交历史
        commit_list = self.get_file_commit_history_cached(file_path, self.commit_count)
        git_history_text = self.format_commit_history_compact(commit_list)

        return {"file_info": file_info, "git_history": git_history_text}

    def extract_file_sections_optimized(self, content: str) -> List[Dict]:
        """
        优化的文件部分提取
        """
        files = []

        # 使用更高效的正则表达式
        file_pattern = r"## File:\s*(.+?)(?=\n)"

        for match in re.finditer(file_pattern, content):
            file_path = match.group(1).strip()

            # 过滤明显无效的路径
            if not file_path or file_path == "path/to/file)" or len(file_path) > 200:
                continue

            start_pos = match.end()

            # 查找下一个文件标题的位置
            next_match = re.search(file_pattern, content[start_pos:])
            end_pos = start_pos + next_match.start() if next_match else len(content)

            # 查找代码块
            file_section = content[start_pos:end_pos]
            code_blocks = list(re.finditer(r"```[\s\S]*?```", file_section))

            if code_blocks:
                last_code_block = code_blocks[-1]
                code_end_absolute = start_pos + last_code_block.end()

                files.append(
                    {
                        "path": file_path,
                        "insert_position": code_end_absolute,
                        "section_content": file_section,
                    }
                )

        self.stats["total_files"] = len(files)
        print(f"📊 发现 {len(files)} 个文件需要处理")
        return files

    def process_markdown_optimized(self, md_content: str) -> str:
        """
        优化的markdown处理
        """
        start_time = time.time()

        print("🔍 分析文档结构...")
        files = self.extract_file_sections_optimized(md_content)

        if not files:
            print("⚠️  未找到需要处理的文件")
            return md_content

        print(f"📋 计划处理 {len(files)} 个文件，使用 {self.max_workers} 个并发线程")

        # 按批次处理文件
        batch_size = max(1, len(files) // self.max_workers)
        processed_results = []

        for i in range(0, len(files), batch_size):
            batch = files[i : i + batch_size]
            print(
                f"🔄 处理批次 {i // batch_size + 1}/{(len(files) + batch_size - 1) // batch_size}"
            )

            batch_results = self.process_file_batch(batch)
            processed_results.extend(batch_results)

        # 应用结果到markdown内容
        print("📝 应用Git历史信息到文档...")

        # 按插入位置从后往前排序，避免位置偏移
        processed_results.sort(
            key=lambda x: x["file_info"]["insert_position"], reverse=True
        )

        for result in processed_results:
            insert_pos = result["file_info"]["insert_position"]
            git_history = result["git_history"]

            # 在指定位置插入Git历史
            md_content = md_content[:insert_pos] + git_history + md_content[insert_pos:]

        self.stats["processing_time"] = time.time() - start_time

        # 打印统计信息
        self._print_stats()

        return md_content

    def _print_stats(self):
        """打印处理统计信息"""
        print("\n" + "=" * 50)
        print("📊 处理统计信息:")
        print(f"📁 总文件数: {self.stats['total_files']}")
        print(f"✅ 已处理: {self.stats['processed_files']}")
        print(f"⏭️  已跳过: {self.stats['skipped_files']}")
        print(f"🎯 缓存命中: {self.stats['cached_hits']}")
        print(f"⏱️  处理时间: {self.stats['processing_time']:.2f} 秒")

        if self.stats["total_files"] > 0:
            rate = (
                self.stats["processed_files"] / self.stats["processing_time"]
                if self.stats["processing_time"] > 0
                else 0
            )
            print(f"🚀 处理速度: {rate:.2f} 文件/秒")
        print("=" * 50)

    def cleanup(self):
        """清理临时目录"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            print(f"🧹 清理临时目录: {self.temp_dir}")
            shutil.rmtree(self.temp_dir)
            self.temp_dir = None

    def __enter__(self):
        """支持上下文管理器"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出时自动清理"""
        self.cleanup()

    def process_file(self, input_file: str, output_file: str = None):
        """
        处理markdown文件（优化版）
        """
        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"输入文件不存在: {input_file}")

        # 检查文件大小
        file_size_mb = input_path.stat().st_size / (1024 * 1024)
        print(f"📄 输入文件大小: {file_size_mb:.2f} MB")

        if file_size_mb > 100:
            print("⚠️  文件较大，处理可能需要较长时间...")
            # 对于超大文件，自动调整参数
            self.commit_count = min(self.commit_count, 2)
            self.max_workers = min(self.max_workers, 2)
            print(
                f"🔧 自动调整参数: 提交数={self.commit_count}, 并发数={self.max_workers}"
            )

        # 验证Git仓库
        if not self.repo_path or not self.repo_path.exists():
            raise FileNotFoundError(f"Git仓库路径不存在: {self.repo_path}")

        try:
            git.Repo(self.repo_path)
            print(f"📦 使用Git仓库: {self.repo_path}")
        except git.exc.InvalidGitRepositoryError:
            raise ValueError(f"指定路径不是有效的Git仓库: {self.repo_path}")

        # 读取原文件
        print("📖 读取文档...")
        with open(input_path, "r", encoding="utf-8") as f:
            content = f.read()

        print(f"🚀 开始处理文件: {input_path}")

        # 处理内容
        processed_content = self.process_markdown_optimized(content)

        # 确定输出文件路径
        if output_file:
            output_path = Path(output_file)
        else:
            # 创建备份
            backup_path = input_path.with_suffix(input_path.suffix + ".backup")
            if not backup_path.exists():  # 避免重复备份
                input_path.rename(backup_path)
                print(f"💾 原文件已备份为: {backup_path}")
            output_path = input_path

        # 写入处理后的内容
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(processed_content)

        print(f"✅ 处理完成，输出文件: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="为repomix生成的markdown文档添加Git提交信息（优化版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  %(prog)s repo.md                              # 处理repo.md文件
  %(prog)s repo.md -o output.md                 # 输出到新文件
  %(prog)s repo.md -r owner/repo                # 指定GitHub远程仓库
  %(prog)s repo.md -c 2                         # 每个文件显示2条提交历史
  %(prog)s repo.md --max-workers 8              # 使用8个并发线程
  %(prog)s repo.md --max-file-size 100          # 跳过超过100MB的文件
  %(prog)s repo.md --skip-pattern "*.log,*.tmp" # 跳过特定文件类型
        """,
    )

    parser.add_argument("input_file", help="输入的markdown文件路径")
    parser.add_argument("-o", "--output", help="输出文件路径（可选，默认覆盖原文件）")
    parser.add_argument("-r", "--repo", help="Git仓库路径或GitHub格式(owner/repo)")
    parser.add_argument(
        "-c",
        "--commit-count",
        type=int,
        default=3,
        help="显示的提交历史数量（默认为3条）",
    )
    parser.add_argument(
        "--max-workers", type=int, default=4, help="最大并发工作线程数（默认为4）"
    )
    parser.add_argument(
        "--max-file-size",
        type=int,
        default=50,
        help="跳过超过此大小的文件（MB，默认50）",
    )
    parser.add_argument("--target-commit", help="目标提交SHA")
    parser.add_argument("--skip-pattern", help="跳过的文件模式，用逗号分隔")

    args = parser.parse_args()

    try:
        # 处理跳过模式
        skip_patterns = None
        if args.skip_pattern:
            skip_patterns = [
                pattern.strip() for pattern in args.skip_pattern.split(",")
            ]

        # 判断是否为远程仓库
        is_remote = False
        if args.repo:
            if args.repo.startswith("http") or (
                "/" in args.repo
                and not args.repo.startswith("/")
                and not os.path.exists(args.repo)
            ):
                is_remote = True

        if is_remote:
            # 使用上下文管理器处理远程仓库
            with OptimizedGitCommitProcessor(
                remote_repo=args.repo,
                target_commit=args.target_commit,
                max_workers=args.max_workers,
                max_file_size_mb=args.max_file_size,
                skip_patterns=skip_patterns,
                commit_count=args.commit_count,
            ) as processor:
                processor.process_file(args.input_file, args.output)
        else:
            # 处理本地仓库
            repo_path = args.repo if args.repo else "."
            processor = OptimizedGitCommitProcessor(
                repo_path=repo_path,
                target_commit=args.target_commit,
                max_workers=args.max_workers,
                max_file_size_mb=args.max_file_size,
                skip_patterns=skip_patterns,
                commit_count=args.commit_count,
            )
            processor.process_file(args.input_file, args.output)

    except Exception as e:
        print(f"❌ 错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
