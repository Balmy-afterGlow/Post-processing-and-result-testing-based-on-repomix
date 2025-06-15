#!/usr/bin/env python3
"""
获取与问题相关的上下文块脚本
基于已构建的向量数据库，为issues.json中的每个问题获取相关上下文
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import logging
from collections import defaultdict

# 从同目录导入RAG系统
from rag_system import LangChainRAGSystem


class RelevantContextRetriever:
    """相关上下文检索器"""

    def __init__(
        self,
        vector_db_dir: str = "./vector_dbs",
        issues_file: str = "../evalulate/datasets/issues.json",
        output_dir: str = "./relevant_context",
    ):
        """
        初始化检索器

        Args:
            vector_db_dir: 向量数据库目录
            issues_file: issues.json文件路径
            output_dir: 相关上下文输出目录
        """
        self.vector_db_dir = Path(vector_db_dir)
        self.issues_file = Path(issues_file)
        self.output_dir = Path(output_dir)

        # 创建输出目录
        self.output_dir.mkdir(exist_ok=True)

        # 设置日志
        self._setup_logging()

        # 加载问题数据
        self.issues = self._load_issues()

        # 按仓库分组问题
        self.issues_by_repo = self._group_issues_by_repo()

        # 统计信息
        self.stats = {
            "total_issues": len(self.issues),
            "total_repos": len(self.issues_by_repo),
            "processed_issues": 0,
            "failed_issues": 0,
            "processing_errors": [],
        }

    def _setup_logging(self):
        """设置日志配置"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler("obtain_relevant_context.log"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def _load_issues(self) -> List[Dict]:
        """加载issues.json文件"""
        if not self.issues_file.exists():
            self.logger.error(f"Issues文件不存在: {self.issues_file}")
            return []

        try:
            with open(self.issues_file, "r", encoding="utf-8") as f:
                issues = json.load(f)

            self.logger.info(f"成功加载 {len(issues)} 个问题")
            return issues
        except Exception as e:
            self.logger.error(f"加载issues文件失败: {e}")
            return []

    def _group_issues_by_repo(self) -> Dict[str, List[Dict]]:
        """按仓库分组问题"""
        grouped = defaultdict(list)

        for issue in self.issues:
            repo = issue.get("repo")
            if repo:
                # 从repo全名提取仓库名 (e.g., "square/okhttp" -> "okhttp")
                repo_name = repo.split("/")[-1] if "/" in repo else repo
                grouped[repo_name].append(issue)

        self.logger.info(f"问题分布在 {len(grouped)} 个仓库中:")
        for repo_name, issues in grouped.items():
            self.logger.info(f"  - {repo_name}: {len(issues)} 个问题")

        return dict(grouped)

    def _construct_query(self, issue: Dict) -> str:
        """构建查询字符串"""
        title = issue.get("issue_title", "")
        body = issue.get("issue_body", "")

        # 拼接标题和正文
        query = ""
        if title:
            query += title
        if body:
            if query:
                query += "\n\n"
            query += body

        return query.strip()

    def _load_rag_system(
        self, repo_name: str, version: str
    ) -> Optional[LangChainRAGSystem]:
        """加载指定仓库和版本的RAG系统"""
        try:
            collection_name = f"{repo_name}_{version}"
            persist_dir = (
                self.vector_db_dir / f"repository-{repo_name}" / f"chroma_db_{version}"
            )

            if not persist_dir.exists():
                self.logger.warning(f"向量数据库不存在: {persist_dir}")
                return None

            rag_system = LangChainRAGSystem(collection_name)
            rag_system.persist_directory = str(persist_dir)

            # 验证是否可以加载
            test_results = rag_system.search("test", k=1)
            if test_results is None:
                self.logger.warning(f"无法加载向量数据库: {persist_dir}")
                return None

            return rag_system

        except Exception as e:
            self.logger.error(f"加载RAG系统失败 {repo_name}/{version}: {e}")
            return None

    def _search_relevant_context(self, query: str, repo_name: str, k: int = 5) -> Dict:
        """搜索相关上下文"""
        results = {}

        # 三种版本的RAG系统
        versions = ["basic", "enhanced", "compressed"]

        for version in versions:
            rag_system = self._load_rag_system(repo_name, version)

            if rag_system:
                try:
                    search_results = rag_system.search(query, k=k)
                    results[version] = search_results
                    self.logger.debug(
                        f"仓库 {repo_name} {version}版本: 找到 {len(search_results)} 个结果"
                    )
                except Exception as e:
                    self.logger.error(f"搜索失败 {repo_name}/{version}: {e}")
                    results[version] = []
            else:
                results[version] = []

        return results

    def process_single_issue(self, issue: Dict, repo_name: str) -> bool:
        """处理单个问题"""
        issue_id = issue.get("id", "unknown")
        issue_title = issue.get("issue_title", "")

        self.logger.info(f"处理问题: {issue_id} - {issue_title[:50]}...")

        try:
            # 构建查询
            query = self._construct_query(issue)

            if not query:
                self.logger.warning(f"问题 {issue_id}: 查询内容为空")
                return False

            # 搜索相关上下文
            search_results = self._search_relevant_context(query, repo_name, k=20)

            # 创建输出目录（在独立的relevant_context目录下）
            output_dir = self.output_dir / issue_id
            output_dir.mkdir(parents=True, exist_ok=True)

            # 保存结果
            output_file = output_dir / "rag_comparison_results.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(search_results, f, ensure_ascii=False, indent=2)

            self.logger.info(f"✅ 问题 {issue_id} 处理完成，结果保存到: {output_file}")
            self.stats["processed_issues"] += 1
            return True

        except Exception as e:
            self.logger.error(f"❌ 问题 {issue_id} 处理失败: {e}")
            self.stats["failed_issues"] += 1
            self.stats["processing_errors"].append(
                {
                    "issue_id": issue_id,
                    "repo_name": repo_name,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                }
            )
            return False

    def process_repo_issues(self, repo_name: str) -> Dict:
        """处理单个仓库的所有问题"""
        issues = self.issues_by_repo.get(repo_name, [])

        if not issues:
            self.logger.warning(f"仓库 {repo_name}: 没有找到问题")
            return {"processed": 0, "failed": 0}

        # 检查向量数据库是否存在
        repo_db_dir = self.vector_db_dir / f"repository-{repo_name}"
        if not repo_db_dir.exists():
            self.logger.warning(f"仓库 {repo_name}: 向量数据库目录不存在")
            return {"processed": 0, "failed": len(issues)}

        self.logger.info(f"开始处理仓库 {repo_name} 的 {len(issues)} 个问题")

        processed = 0
        failed = 0

        for issue in issues:
            if self.process_single_issue(issue, repo_name):
                processed += 1
            else:
                failed += 1

        self.logger.info(f"仓库 {repo_name} 处理完成: 成功 {processed}, 失败 {failed}")
        return {"processed": processed, "failed": failed}

    def process_all_issues(self) -> Dict:
        """处理所有问题"""
        self.logger.info("开始处理所有问题...")

        if not self.vector_db_dir.exists():
            self.logger.error(f"向量数据库目录不存在: {self.vector_db_dir}")
            return self.stats

        repo_results = {}

        for repo_name in self.issues_by_repo.keys():
            self.logger.info(f"处理仓库: {repo_name}")
            repo_results[repo_name] = self.process_repo_issues(repo_name)

        # 生成总结报告
        self.logger.info("=" * 60)
        self.logger.info("处理完成! 统计报告:")
        self.logger.info(f"总问题数: {self.stats['total_issues']}")
        self.logger.info(f"总仓库数: {self.stats['total_repos']}")
        self.logger.info(f"成功处理: {self.stats['processed_issues']}")
        self.logger.info(f"失败问题: {self.stats['failed_issues']}")

        if self.stats["processing_errors"]:
            self.logger.info("\n失败的问题:")
            for error in self.stats["processing_errors"]:
                self.logger.info(
                    f"  - {error['issue_id']} ({error['repo_name']}): {error['error']}"
                )

        # 保存统计信息
        stats_file = self.output_dir / "context_retrieval_statistics.json"
        final_stats = {
            **self.stats,
            "repo_results": repo_results,
            "completion_time": datetime.now().isoformat(),
        }

        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(final_stats, f, ensure_ascii=False, indent=2)

        self.logger.info(f"统计信息已保存到: {stats_file}")
        return final_stats


def main():
    """主函数"""
    print("=" * 60)
    print("🔍 相关上下文检索工具")
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
        print("请确保rag_system.py在同一目录下，并已构建向量数据库")
        return

    # 创建检索器
    retriever = RelevantContextRetriever()

    # 检查数据文件
    if not retriever.issues:
        print("❌ 没有找到有效的问题数据")
        return

    if not retriever.issues_by_repo:
        print("❌ 没有找到任何仓库问题")
        return

    print(
        f"📊 待处理: {retriever.stats['total_issues']} 个问题，分布在 {retriever.stats['total_repos']} 个仓库中"
    )

    # 执行处理
    stats = retriever.process_all_issues()

    print("\n" + "=" * 60)
    print("🎉 处理完成!")
    print(f"📊 成功: {stats['processed_issues']}, 失败: {stats['failed_issues']}")

    print("\n📁 输出目录结构:")
    output_dir = Path("./relevant_context")
    if output_dir.exists():
        for repo_dir in sorted(output_dir.iterdir()):
            if repo_dir.is_dir():
                print(f"  📂 {repo_dir.name}/")
                issue_dirs = [d for d in repo_dir.iterdir() if d.is_dir()]
                for issue_dir in sorted(issue_dirs)[:3]:  # 只显示前3个
                    print(f"    📁 {issue_dir.name}/")
                    print("      📄 rag_comparison_results.json")
                if len(issue_dirs) > 3:
                    print(f"    ... 和其他 {len(issue_dirs) - 3} 个问题目录")

    print("\n📝 日志文件: obtain_relevant_context.log")
    print("📈 统计文件: ./relevant_context/context_retrieval_statistics.json")


if __name__ == "__main__":
    main()
