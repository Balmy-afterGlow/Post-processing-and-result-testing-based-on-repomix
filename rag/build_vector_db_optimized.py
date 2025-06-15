#!/usr/bin/env python3
"""
优化版批量向量数据库构建脚本
针对大型仓库进行了性能优化：文档过滤、批处理、智能采样
"""

import os
import json
import shutil
from pathlib import Path
from typing import List, Dict
from datetime import datetime
import logging

# 设置环境变量以强制使用本地缓存，避免网络请求
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# 从同目录导入RAG系统
from rag_system import MarkdownParser, DocumentProcessor, LangChainRAGSystem


class OptimizedVectorDatabaseBuilder:
    """优化版向量数据库批量构建器"""

    def __init__(
        self,
        repomix_dir: str = "../repomix_md",
        output_dir: str = "./vector_dbs_optimized",
        max_documents_per_version: int = 2000,  # 每个版本最大文档数
        min_content_length: int = 100,  # 最小内容长度
        batch_size: int = 100,  # 批处理大小
        skip_large_files: bool = True,  # 跳过大文件
        max_file_size_mb: float = 5.0,  # 最大文件大小(MB)
    ):
        """
        初始化优化构建器

        Args:
            repomix_dir: repomix输出目录
            output_dir: 向量数据库输出目录
            max_documents_per_version: 每个版本最大文档数量
            min_content_length: 最小内容长度过滤
            batch_size: 批处理大小
            skip_large_files: 是否跳过大文件
            max_file_size_mb: 最大处理文件大小(MB)
        """
        self.repomix_dir = Path(repomix_dir)
        self.output_dir = Path(output_dir)
        self.max_documents_per_version = max_documents_per_version
        self.min_content_length = min_content_length
        self.batch_size = batch_size
        self.skip_large_files = skip_large_files
        self.max_file_size_mb = max_file_size_mb

        self.parser = MarkdownParser()
        self.processor = DocumentProcessor()

        # 创建输出目录
        self.output_dir.mkdir(exist_ok=True)

        # 设置日志
        self._setup_logging()

        # 统计信息
        self.stats = {
            "total_repos": 0,
            "processed_repos": 0,
            "skipped_repos": 0,
            "failed_repos": 0,
            "total_documents": 0,
            "filtered_documents": 0,
            "processing_errors": [],
        }

    def _setup_logging(self):
        """设置日志配置"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler("optimized_vector_db_build.log"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def filter_and_prioritize_documents(
        self, documents: List, version_name: str
    ) -> List:
        """
        过滤和优先排序文档

        Args:
            documents: 原始文档列表
            version_name: 版本名称

        Returns:
            过滤和排序后的文档列表
        """
        # 1. 按内容长度过滤
        filtered_docs = [
            doc
            for doc in documents
            if len(doc.page_content.strip()) >= self.min_content_length
        ]

        original_count = len(documents)
        after_filter_count = len(filtered_docs)

        self.logger.info(
            f"{version_name}: 长度过滤 {original_count} -> {after_filter_count} 个文档"
        )

        # 2. 按重要性排序
        def get_document_priority(doc):
            priority = 0
            content = doc.page_content.lower()
            metadata = doc.metadata

            # 重要文件类型优先
            important_extensions = [
                ".py",
                ".js",
                ".ts",
                ".java",
                ".cpp",
                ".c",
                ".go",
                ".rs",
            ]
            if any(
                ext in metadata.get("file_path", "").lower()
                for ext in important_extensions
            ):
                priority += 20

            # 配置文件和重要文件
            important_files = ["readme", "config", "setup", "main", "index", "__init__"]
            if any(
                name in metadata.get("file_path", "").lower()
                for name in important_files
            ):
                priority += 15

            # 代码结构关键词
            code_keywords = [
                "class ",
                "function ",
                "def ",
                "import ",
                "from ",
                "export",
            ]
            priority += sum(3 for keyword in code_keywords if keyword in content)

            # 避免测试文件过多
            if "test" in metadata.get("file_path", "").lower():
                priority -= 5

            # 内容长度适中优先
            content_len = len(doc.page_content)
            if 200 <= content_len <= 2000:
                priority += 10
            elif content_len > 5000:
                priority -= 5

            return priority

        # 3. 排序并限制数量
        filtered_docs.sort(key=get_document_priority, reverse=True)

        if len(filtered_docs) > self.max_documents_per_version:
            filtered_docs = filtered_docs[: self.max_documents_per_version]
            self.logger.info(
                f"{version_name}: 数量限制 {after_filter_count} -> {len(filtered_docs)} 个文档"
            )

        self.stats["filtered_documents"] += original_count - len(filtered_docs)
        return filtered_docs

    def build_vector_db_in_batches(
        self, documents: List, rag_system, version_name: str
    ):
        """
        批量构建向量数据库

        Args:
            documents: 文档列表
            rag_system: RAG系统实例
            version_name: 版本名称
        """
        total_docs = len(documents)
        if total_docs == 0:
            self.logger.warning(f"{version_name}: 没有文档需要索引")
            return

        self.logger.info(
            f"{version_name}: 开始批量索引 {total_docs} 个文档，批大小: {self.batch_size}"
        )

        # 分批处理
        for i in range(0, total_docs, self.batch_size):
            batch = documents[i : i + self.batch_size]
            batch_num = i // self.batch_size + 1
            total_batches = (total_docs + self.batch_size - 1) // self.batch_size

            self.logger.info(
                f"{version_name}: 处理批次 {batch_num}/{total_batches} ({len(batch)} 个文档)"
            )

            try:
                if i == 0:
                    # 第一批：创建新的向量存储
                    rag_system.vectorstore = (
                        rag_system._create_vectorstore_from_documents(batch)
                    )
                else:
                    # 后续批次：添加到现有向量存储
                    rag_system._add_documents_to_vectorstore(batch)

            except Exception as e:
                self.logger.error(f"{version_name}: 批次 {batch_num} 索引失败: {e}")
                raise

    def check_file_size(self, file_path: str) -> bool:
        """检查文件大小是否合适处理"""
        try:
            file_size_mb = Path(file_path).stat().st_size / (1024 * 1024)
            if self.skip_large_files and file_size_mb > self.max_file_size_mb:
                self.logger.warning(f"跳过大文件 {file_path}: {file_size_mb:.1f}MB")
                return False
            return True
        except Exception:
            return True  # 如果无法获取文件大小，继续处理

    def find_git_enhanced_files(self) -> List[Dict[str, str]]:
        """查找所有的Git增强版markdown文件"""
        git_files = []

        if not self.repomix_dir.exists():
            self.logger.error(f"repomix目录不存在: {self.repomix_dir}")
            return git_files

        for repo_dir in self.repomix_dir.iterdir():
            if not repo_dir.is_dir() or not repo_dir.name.startswith("repository-"):
                continue

            # 提取仓库名
            repo_name = repo_dir.name.replace("repository-", "")

            # 查找Git增强版文件
            git_file_pattern = f"repomix-output-{repo_name}-with-git.md"
            git_file_path = repo_dir / git_file_pattern

            if git_file_path.exists():
                # 检查文件大小，给出预估时间
                file_size_mb = git_file_path.stat().st_size / (1024 * 1024)

                git_files.append(
                    {
                        "repo_name": repo_name,
                        "repo_dir": str(repo_dir),
                        "git_file": str(git_file_path),
                        "file_size_mb": file_size_mb,
                        "output_dir": str(self.output_dir / f"repository-{repo_name}"),
                    }
                )
                self.logger.info(f"找到Git增强文件: {repo_name} ({file_size_mb:.1f}MB)")
            else:
                self.logger.warning(f"跳过仓库 {repo_name}: 未找到Git增强版文件")
                self.stats["skipped_repos"] += 1

        self.stats["total_repos"] = len(git_files) + self.stats["skipped_repos"]
        self.logger.info(f"总共找到 {len(git_files)} 个Git增强版文件")

        # 按文件大小排序，先处理小文件
        git_files.sort(key=lambda x: x["file_size_mb"])
        return git_files

    def build_single_repository(self, repo_info: Dict[str, str]) -> bool:
        """为单个仓库构建向量数据库（优化版）"""
        repo_name = repo_info["repo_name"]
        git_file = repo_info["git_file"]
        output_dir = repo_info["output_dir"]
        file_size_mb = repo_info.get("file_size_mb", 0)

        self.logger.info(f"开始处理仓库: {repo_name} ({file_size_mb:.1f}MB)")

        # 检查文件大小
        if not self.check_file_size(git_file):
            self.logger.warning(f"跳过仓库 {repo_name}: 文件过大")
            self.stats["skipped_repos"] += 1
            return False

        try:
            # 检查输出目录是否已存在
            if os.path.exists(output_dir):
                self.logger.info(f"清理已存在的向量数据库: {repo_name}")
                shutil.rmtree(output_dir)

            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)

            # 读取markdown文件
            with open(git_file, "r", encoding="utf-8") as f:
                md_content = f.read()

            # 解析代码块
            start_time = datetime.now()
            code_blocks = self.parser.parse_markdown(md_content)
            parse_time = (datetime.now() - start_time).total_seconds()

            if not code_blocks:
                self.logger.warning(f"仓库 {repo_name}: 未解析到任何代码块")
                return False

            self.logger.info(
                f"仓库 {repo_name}: 解析出 {len(code_blocks)} 个代码块 (耗时 {parse_time:.1f}s)"
            )

            # 创建三种不同版本的文档
            basic_docs = self.processor.create_basic_documents(code_blocks)
            enhanced_docs = self.processor.create_enhanced_documents(code_blocks)
            compressed_docs = self.processor.create_compressed_documents(code_blocks)

            self.logger.info(
                f"仓库 {repo_name}: 原始文档数 - 基础版:{len(basic_docs)}, 增强版:{len(enhanced_docs)}, 压缩版:{len(compressed_docs)}"
            )

            # 过滤和优化文档
            filtered_basic = self.filter_and_prioritize_documents(
                basic_docs, f"{repo_name}_basic"
            )
            filtered_enhanced = self.filter_and_prioritize_documents(
                enhanced_docs, f"{repo_name}_enhanced"
            )
            filtered_compressed = self.filter_and_prioritize_documents(
                compressed_docs, f"{repo_name}_compressed"
            )

            # 为每种版本创建向量数据库
            versions = [
                ("basic", filtered_basic),
                ("enhanced", filtered_enhanced),
                ("compressed", filtered_compressed),
            ]

            index_start_time = datetime.now()

            for version_name, documents in versions:
                if not documents:
                    self.logger.warning(
                        f"仓库 {repo_name} {version_name}版本: 过滤后没有文档需要索引"
                    )
                    continue

                # 创建RAG系统
                collection_name = f"{repo_name}_{version_name}"
                persist_dir = os.path.join(output_dir, f"chroma_db_{version_name}")

                rag_system = LangChainRAGSystem(collection_name)
                rag_system.persist_directory = persist_dir

                # 批量索引文档
                version_start_time = datetime.now()
                self.build_vector_db_in_batches(
                    documents, rag_system, f"{repo_name}_{version_name}"
                )
                version_time = (datetime.now() - version_start_time).total_seconds()

                self.logger.info(
                    f"仓库 {repo_name} {version_name}版本: 已索引 {len(documents)} 个文档 (耗时 {version_time:.1f}s)"
                )

            total_index_time = (datetime.now() - index_start_time).total_seconds()

            # 保存仓库元数据
            metadata = {
                "repo_name": repo_name,
                "source_file": git_file,
                "file_size_mb": file_size_mb,
                "build_time": datetime.now().isoformat(),
                "processing_time": {
                    "parse_time_seconds": parse_time,
                    "index_time_seconds": total_index_time,
                    "total_time_seconds": parse_time + total_index_time,
                },
                "code_blocks_count": len(code_blocks),
                "original_documents_count": {
                    "basic": len(basic_docs),
                    "enhanced": len(enhanced_docs),
                    "compressed": len(compressed_docs),
                },
                "filtered_documents_count": {
                    "basic": len(filtered_basic),
                    "enhanced": len(filtered_enhanced),
                    "compressed": len(filtered_compressed),
                },
                "optimization_settings": {
                    "max_documents_per_version": self.max_documents_per_version,
                    "min_content_length": self.min_content_length,
                    "batch_size": self.batch_size,
                    "max_file_size_mb": self.max_file_size_mb,
                },
                "total_commits": sum(len(block.git_commits) for block in code_blocks),
                "files_processed": [block.file_path for block in code_blocks],
            }

            metadata_file = os.path.join(output_dir, "metadata.json")
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            self.stats["processed_repos"] += 1
            self.stats["total_documents"] += (
                len(filtered_basic) + len(filtered_enhanced) + len(filtered_compressed)
            )

            self.logger.info(
                f"✅ 仓库 {repo_name} 处理完成 (总耗时 {parse_time + total_index_time:.1f}s)"
            )
            return True

        except Exception as e:
            self.logger.error(f"❌ 仓库 {repo_name} 处理失败: {e}")
            self.stats["failed_repos"] += 1
            self.stats["processing_errors"].append(
                {
                    "repo_name": repo_name,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                }
            )
            return False

    def build_all_repositories(self) -> Dict:
        """批量构建所有仓库的向量数据库"""
        self.logger.info("开始批量构建向量数据库（优化版）...")
        self.logger.info(
            f"优化设置: 最大文档数={self.max_documents_per_version}, 批大小={self.batch_size}"
        )

        # 查找所有Git增强版文件
        git_files = self.find_git_enhanced_files()

        if not git_files:
            self.logger.error("没有找到任何Git增强版文件")
            return self.stats

        # 估算总处理时间
        total_size_mb = sum(f.get("file_size_mb", 0) for f in git_files)
        self.logger.info(f"预计处理 {total_size_mb:.1f}MB 数据")

        # 处理每个仓库
        start_time = datetime.now()
        for i, repo_info in enumerate(git_files, 1):
            self.logger.info(
                f"进度: {i}/{len(git_files)} - 处理仓库 {repo_info['repo_name']}"
            )

            if self.build_single_repository(repo_info):
                # 计算平均速度和剩余时间
                elapsed_time = (datetime.now() - start_time).total_seconds()
                avg_time_per_repo = elapsed_time / i
                remaining_repos = len(git_files) - i
                estimated_remaining_time = avg_time_per_repo * remaining_repos

                self.logger.info(
                    f"平均处理时间: {avg_time_per_repo:.1f}s/仓库, 预计剩余时间: {estimated_remaining_time / 60:.1f}分钟"
                )

        total_time = (datetime.now() - start_time).total_seconds()

        # 生成总结报告
        self.logger.info("=" * 60)
        self.logger.info("构建完成! 统计报告:")
        self.logger.info(f"总仓库数: {self.stats['total_repos']}")
        self.logger.info(f"成功处理: {self.stats['processed_repos']}")
        self.logger.info(f"跳过仓库: {self.stats['skipped_repos']}")
        self.logger.info(f"失败仓库: {self.stats['failed_repos']}")
        self.logger.info(f"总文档数: {self.stats['total_documents']}")
        self.logger.info(f"过滤文档数: {self.stats['filtered_documents']}")
        self.logger.info(f"总处理时间: {total_time / 60:.1f}分钟")
        self.logger.info(
            f"平均处理速度: {total_time / max(1, self.stats['processed_repos']):.1f}s/仓库"
        )

        if self.stats["processing_errors"]:
            self.logger.info("\n失败的仓库:")
            for error in self.stats["processing_errors"]:
                self.logger.info(f"  - {error['repo_name']}: {error['error']}")

        # 保存统计信息
        self.stats["total_processing_time_seconds"] = total_time
        stats_file = self.output_dir / "build_statistics.json"
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2)

        self.logger.info(f"统计信息已保存到: {stats_file}")
        return self.stats

    def verify_builds(self) -> Dict[str, bool]:
        """验证构建结果"""
        self.logger.info("开始验证构建结果...")

        verification_results = {}

        for repo_dir in self.output_dir.iterdir():
            if not repo_dir.is_dir() or not repo_dir.name.startswith("repository-"):
                continue

            repo_name = repo_dir.name.replace("repository-", "")

            # 检查必需文件
            required_files = [
                "metadata.json",
                "chroma_db_basic",
                "chroma_db_enhanced",
                "chroma_db_compressed",
            ]

            all_exists = True
            for req_file in required_files:
                file_path = repo_dir / req_file
                if not file_path.exists():
                    self.logger.warning(f"仓库 {repo_name}: 缺少 {req_file}")
                    all_exists = False

            verification_results[repo_name] = all_exists

            if all_exists:
                self.logger.info(f"✅ 仓库 {repo_name}: 验证通过")
            else:
                self.logger.error(f"❌ 仓库 {repo_name}: 验证失败")

        # 保存验证结果
        verification_file = self.output_dir / "verification_results.json"
        with open(verification_file, "w", encoding="utf-8") as f:
            json.dump(verification_results, f, ensure_ascii=False, indent=2)

        passed = sum(verification_results.values())
        total = len(verification_results)
        self.logger.info(f"验证完成: {passed}/{total} 个仓库通过验证")

        return verification_results


def main():
    """主函数"""
    print("=" * 60)
    print("🚀 优化版批量向量数据库构建工具")
    print("=" * 60)

    # 检查依赖
    try:
        import importlib.util

        spec = importlib.util.find_spec("rag_system")
        if spec is None:
            raise ImportError("rag_system module not found")
        print("✅ 依赖检查通过")
    except ImportError as e:
        print(f"❌ 依赖检查失败: {e}")
        print("请确保rag_system.py在同一目录下，并安装了所需依赖")
        return

    # 创建构建器（优化参数）
    builder = OptimizedVectorDatabaseBuilder(
        max_documents_per_version=2000,  # 从18k降到2k
        min_content_length=100,
        batch_size=100,
        skip_large_files=True,
        max_file_size_mb=5.0,
    )

    print(f"🔧 优化设置:")
    print(f"   最大文档数/版本: {builder.max_documents_per_version}")
    print(f"   最小内容长度: {builder.min_content_length}")
    print(f"   批处理大小: {builder.batch_size}")
    print(f"   最大文件大小: {builder.max_file_size_mb}MB")
    print()

    # 执行构建
    stats = builder.build_all_repositories()

    # 验证结果
    verification = builder.verify_builds()

    print("\n" + "=" * 60)
    print("🎉 构建完成!")
    print(
        f"📊 成功: {stats['processed_repos']}, 失败: {stats['failed_repos']}, 跳过: {stats['skipped_repos']}"
    )
    print(f"📄 总文档数: {stats['total_documents']}")
    print(f"🗂️ 过滤文档数: {stats['filtered_documents']}")
    print(f"⏱️ 总处理时间: {stats.get('total_processing_time_seconds', 0) / 60:.1f}分钟")

    passed_verification = sum(verification.values())
    total_verification = len(verification)
    print(f"✅ 验证通过: {passed_verification}/{total_verification}")

    print("\n📁 输出目录结构:")
    output_dir = Path("./vector_dbs_optimized")
    if output_dir.exists():
        for item in sorted(output_dir.iterdir()):
            if item.is_dir() and item.name.startswith("repository-"):
                print(f"  📂 {item.name}/")
                for subitem in sorted(item.iterdir()):
                    if subitem.is_dir():
                        print(f"    📊 {subitem.name}/")
                    else:
                        print(f"    📄 {subitem.name}")

    print("\n📝 日志文件: optimized_vector_db_build.log")
    print("📈 统计文件: ./vector_dbs_optimized/build_statistics.json")
    print("🔍 验证文件: ./vector_dbs_optimized/verification_results.json")


if __name__ == "__main__":
    main()
