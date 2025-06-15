#!/usr/bin/env python3
"""
批量向量数据库构建脚本
基于repomix_md目录下的Git增强版markdown文档，为每个仓库构建向量数据库
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


class VectorDatabaseBuilder:
    """向量数据库批量构建器"""

    def __init__(
        self, repomix_dir: str = "../repomix_md", output_dir: str = "./vector_dbs"
    ):
        """
        初始化构建器

        Args:
            repomix_dir: repomix输出目录
            output_dir: 向量数据库输出目录
        """
        self.repomix_dir = Path(repomix_dir)
        self.output_dir = Path(output_dir)
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
            "processing_errors": [],
        }

    def _setup_logging(self):
        """设置日志配置"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler("vector_db_build.log"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

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
                git_files.append(
                    {
                        "repo_name": repo_name,
                        "repo_dir": str(repo_dir),
                        "git_file": str(git_file_path),
                        "output_dir": str(self.output_dir / f"repository-{repo_name}"),
                    }
                )
                self.logger.info(f"找到Git增强文件: {repo_name}")
            else:
                self.logger.warning(f"跳过仓库 {repo_name}: 未找到Git增强版文件")
                self.stats["skipped_repos"] += 1

        self.stats["total_repos"] = len(git_files) + self.stats["skipped_repos"]
        self.logger.info(f"总共找到 {len(git_files)} 个Git增强版文件")
        return git_files

    def build_single_repository(self, repo_info: Dict[str, str]) -> bool:
        """为单个仓库构建向量数据库"""
        repo_name = repo_info["repo_name"]
        git_file = repo_info["git_file"]
        output_dir = repo_info["output_dir"]

        self.logger.info(f"开始处理仓库: {repo_name}")

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
            code_blocks = self.parser.parse_markdown(md_content)

            if not code_blocks:
                self.logger.warning(f"仓库 {repo_name}: 未解析到任何代码块")
                return False

            self.logger.info(f"仓库 {repo_name}: 解析出 {len(code_blocks)} 个代码块")

            # 创建三种不同版本的文档
            basic_docs = self.processor.create_basic_documents(code_blocks)
            enhanced_docs = self.processor.create_enhanced_documents(code_blocks)
            compressed_docs = self.processor.create_compressed_documents(code_blocks)

            self.logger.info(
                f"仓库 {repo_name}: 创建文档 - 基础版:{len(basic_docs)}, 增强版:{len(enhanced_docs)}, 压缩版:{len(compressed_docs)}"
            )

            # 为每种版本创建向量数据库
            versions = [
                ("basic", basic_docs),
                ("enhanced", enhanced_docs),
                ("compressed", compressed_docs),
            ]

            for version_name, documents in versions:
                if not documents:
                    self.logger.warning(
                        f"仓库 {repo_name} {version_name}版本: 没有文档需要索引"
                    )
                    continue

                # 创建RAG系统
                collection_name = f"{repo_name}_{version_name}"
                persist_dir = os.path.join(output_dir, f"chroma_db_{version_name}")

                rag_system = LangChainRAGSystem(collection_name)
                rag_system.persist_directory = persist_dir

                # 索引文档
                rag_system.index_documents(documents)
                self.logger.info(
                    f"仓库 {repo_name} {version_name}版本: 已索引 {len(documents)} 个文档"
                )

            # 保存仓库元数据
            metadata = {
                "repo_name": repo_name,
                "source_file": git_file,
                "build_time": datetime.now().isoformat(),
                "code_blocks_count": len(code_blocks),
                "documents_count": {
                    "basic": len(basic_docs),
                    "enhanced": len(enhanced_docs),
                    "compressed": len(compressed_docs),
                },
                "total_commits": sum(len(block.git_commits) for block in code_blocks),
                "files_processed": [block.file_path for block in code_blocks],
            }

            metadata_file = os.path.join(output_dir, "metadata.json")
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            self.stats["processed_repos"] += 1
            self.stats["total_documents"] += (
                len(basic_docs) + len(enhanced_docs) + len(compressed_docs)
            )

            self.logger.info(f"✅ 仓库 {repo_name} 处理完成")
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
        self.logger.info("开始批量构建向量数据库...")

        # 查找所有Git增强版文件
        git_files = self.find_git_enhanced_files()

        if not git_files:
            self.logger.error("没有找到任何Git增强版文件")
            return self.stats

        # 处理每个仓库
        success_count = 0
        for i, repo_info in enumerate(git_files, 1):
            self.logger.info(
                f"进度: {i}/{len(git_files)} - 处理仓库 {repo_info['repo_name']}"
            )

            if self.build_single_repository(repo_info):
                success_count += 1

        # 生成总结报告
        self.logger.info("=" * 60)
        self.logger.info("构建完成! 统计报告:")
        self.logger.info(f"总仓库数: {self.stats['total_repos']}")
        self.logger.info(f"成功处理: {self.stats['processed_repos']}")
        self.logger.info(f"跳过仓库: {self.stats['skipped_repos']}")
        self.logger.info(f"失败仓库: {self.stats['failed_repos']}")
        self.logger.info(f"总文档数: {self.stats['total_documents']}")

        if self.stats["processing_errors"]:
            self.logger.info("\n失败的仓库:")
            for error in self.stats["processing_errors"]:
                self.logger.info(f"  - {error['repo_name']}: {error['error']}")

        # 保存统计信息
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
    print("🚀 批量向量数据库构建工具")
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

    # 创建构建器
    builder = VectorDatabaseBuilder()

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

    passed_verification = sum(verification.values())
    total_verification = len(verification)
    print(f"✅ 验证通过: {passed_verification}/{total_verification}")

    print("\n📁 输出目录结构:")
    output_dir = Path("./vector_dbs")
    if output_dir.exists():
        for item in sorted(output_dir.iterdir()):
            if item.is_dir() and item.name.startswith("repository-"):
                print(f"  📂 {item.name}/")
                for subitem in sorted(item.iterdir()):
                    if subitem.is_dir():
                        print(f"    📊 {subitem.name}/")
                    else:
                        print(f"    📄 {subitem.name}")

    print("\n📝 日志文件: vector_db_build.log")
    print("📈 统计文件: ./vector_dbs/build_statistics.json")
    print("🔍 验证文件: ./vector_dbs/verification_results.json")


if __name__ == "__main__":
    main()
