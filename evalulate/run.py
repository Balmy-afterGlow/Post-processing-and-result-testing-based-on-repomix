#!/usr/bin/env python3
"""
RAG评估运行脚本
向DeepSeek-V3 LLM输入问题并获取输出，保存结果
"""

import json
import os
import sys
from dotenv import load_dotenv
from pathlib import Path
from typing import Dict, List, Optional
import logging
from datetime import datetime
import time

# 添加当前目录到Python路径
sys.path.append(str(Path(__file__).parent))

from utils.context_composer import compose_chat_with_rag
from utils.get_json_from_ai import extract_json_from_ai_response

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("rag_evaluation.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

load_dotenv()


class RAGEvaluationRunner:
    """RAG评估运行器"""

    def __init__(
        self,
        issues_file: str = "./datasets/issues.json",
        rag_context_dir: str = "../rag/relevant_context",
        output_dir: str = "./results/based_on_rag",
        api_key: Optional[str] = None,
        model_name: str = "deepseek-chat",
        api_base: str = "https://api.deepseek.com",
    ):
        """
        初始化评估运行器

        Args:
            issues_file: issues.json文件路径
            rag_context_dir: RAG上下文目录
            output_dir: 输出结果目录
            api_key: DeepSeek API密钥
            model_name: 模型名称
            api_base: API基础URL
        """
        self.issues_file = Path(issues_file)
        self.rag_context_dir = Path(rag_context_dir)
        self.output_dir = Path(output_dir)
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.model_name = model_name
        self.api_base = api_base

        # 创建输出目录
        for strategy in ["basic", "enhanced", "compressed"]:
            (self.output_dir / strategy).mkdir(parents=True, exist_ok=True)

        # 加载问题数据
        self.issues = self._load_issues()

        # 初始化API客户端
        self._init_api_client()

        # 统计信息
        self.stats = {
            "total_issues": len(self.issues),
            "processed_issues": 0,
            "failed_issues": 0,
            "strategy_stats": {
                "basic": {"success": 0, "failed": 0},
                "enhanced": {"success": 0, "failed": 0},
                "compressed": {"success": 0, "failed": 0},
            },
            "processing_errors": [],
            "start_time": datetime.now().isoformat(),
        }

    def _load_issues(self) -> List[Dict]:
        """加载issues.json文件"""
        if not self.issues_file.exists():
            logger.error(f"Issues文件不存在: {self.issues_file}")
            return []

        try:
            with open(self.issues_file, "r", encoding="utf-8") as f:
                issues = json.load(f)
            logger.info(f"成功加载 {len(issues)} 个问题")
            return issues
        except Exception as e:
            logger.error(f"加载issues文件失败: {e}")
            return []

    def _init_api_client(self):
        """初始化API客户端"""
        if not self.api_key:
            logger.error("未设置DEEPSEEK_API_KEY环境变量")
            raise ValueError("请设置DEEPSEEK_API_KEY环境变量")

        try:
            from openai import OpenAI

            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
            )
            logger.info("DeepSeek API客户端初始化成功")
        except ImportError:
            logger.error("请安装openai包: pip install openai")
            raise
        except Exception as e:
            logger.error(f"API客户端初始化失败: {e}")
            raise

    def _get_repo_name(self, issue: Dict) -> str:
        """从issue中提取仓库名"""
        repo = issue.get("repo", "")
        if "/" in repo:
            return repo.split("/")[-1]
        return repo

    def _find_rag_context_file(self, repo_name: str, issue_id: str) -> Optional[Path]:
        """查找RAG上下文文件"""
        # 尝试不同的目录命名模式
        possible_dirs = [
            self.rag_context_dir / issue_id,
            self.rag_context_dir / repo_name / issue_id,
            self.rag_context_dir / f"repository-{repo_name}" / issue_id,
        ]

        for dir_path in possible_dirs:
            rag_file = dir_path / "rag_comparison_results.json"
            if rag_file.exists():
                return rag_file

        return None

    def _call_deepseek_api(
        self, messages: List[Dict], max_retries: int = 3
    ) -> Optional[str]:
        """调用DeepSeek API"""
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=1000,
                )

                if response.choices and len(response.choices) > 0:
                    return response.choices[0].message.content

                logger.warning(f"API返回空响应，尝试 {attempt + 1}/{max_retries}")

            except Exception as e:
                logger.warning(f"API调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)  # 指数退避
                else:
                    logger.error(f"API调用最终失败: {e}")

        return None

    def _process_single_issue_strategy(
        self, issue: Dict, strategy: str
    ) -> Optional[Dict]:
        """处理单个问题的单个策略"""
        issue_id = issue.get("id", "unknown")
        repo_name = self._get_repo_name(issue)

        logger.info(f"处理问题 {issue_id} - 策略: {strategy}")

        # 查找RAG上下文文件
        rag_file = self._find_rag_context_file(repo_name, issue_id)
        if not rag_file:
            logger.warning(f"未找到问题 {issue_id} 的RAG上下文文件")
            return None

        try:
            # 构造消息
            messages = compose_chat_with_rag(issue, str(rag_file), strategy)

            # 调用API
            response = self._call_deepseek_api(messages)
            if not response:
                logger.error(f"问题 {issue_id} 策略 {strategy}: API调用失败")
                return None

            # 提取JSON
            result_json = extract_json_from_ai_response(response)

            logger.info(f"问题 {issue_id} 策略 {strategy}: 处理成功")
            return result_json

        except Exception as e:
            logger.error(f"问题 {issue_id} 策略 {strategy}: 处理失败 - {e}")
            return None

    def _save_result(self, issue_id: str, strategy: str, result: Dict) -> bool:
        """保存结果到文件"""
        try:
            # 创建输出目录
            output_file = (
                self.output_dir / strategy / f"{issue_id}" / f"{issue_id}.json"
            )
            output_file.parent.mkdir(parents=True, exist_ok=True)

            # 保存结果
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            logger.debug(f"结果已保存到: {output_file}")
            return True

        except Exception as e:
            logger.error(f"保存结果失败 {issue_id}/{strategy}: {e}")
            return False

    def process_single_issue(self, issue: Dict) -> Dict:
        """处理单个问题的所有策略"""
        issue_id = issue.get("id", "unknown")
        issue_title = issue.get("issue_title", "")

        logger.info(f"开始处理问题: {issue_id} - {issue_title[:50]}...")

        results = {}
        strategies = ["basic", "enhanced", "compressed"]

        for strategy in strategies:
            # 检查是否已存在结果文件
            output_file = self.output_dir / strategy / f"{issue_id}.json"
            if output_file.exists():
                logger.info(f"问题 {issue_id} 策略 {strategy}: 结果已存在，跳过")
                self.stats["strategy_stats"][strategy]["success"] += 1
                continue

            # 处理策略
            result = self._process_single_issue_strategy(issue, strategy)

            if result:
                # 保存结果
                if self._save_result(issue_id, strategy, result):
                    self.stats["strategy_stats"][strategy]["success"] += 1
                    results[strategy] = result
                else:
                    self.stats["strategy_stats"][strategy]["failed"] += 1
            else:
                self.stats["strategy_stats"][strategy]["failed"] += 1
                self.stats["processing_errors"].append(
                    {
                        "issue_id": issue_id,
                        "strategy": strategy,
                        "error": "Processing failed",
                        "timestamp": datetime.now().isoformat(),
                    }
                )

            # 添加延时避免API限制
            time.sleep(1)

        self.stats["processed_issues"] += 1
        return results

    def run_all(self, start_index: int = 0, max_issues: Optional[int] = None):
        """运行评估"""
        logger.info("开始RAG评估...")
        logger.info(f"总问题数: {len(self.issues)}")
        logger.info(f"开始索引: {start_index}")

        if max_issues:
            logger.info(f"最大处理数: {max_issues}")

        # 处理问题子集
        issues_to_process = self.issues[start_index:]
        if max_issues:
            issues_to_process = issues_to_process[:max_issues]

        logger.info(f"实际处理问题数: {len(issues_to_process)}")

        start_time = time.time()

        for i, issue in enumerate(issues_to_process):
            current_index = start_index + i
            logger.info(
                f"进度: {i + 1}/{len(issues_to_process)} (总体: {current_index + 1}/{len(self.issues)})"
            )

            try:
                self.process_single_issue(issue)

                # 每10个问题打印一次统计
                if (i + 1) % 10 == 0:
                    self._print_progress_stats()

            except KeyboardInterrupt:
                logger.info("收到中断信号，正在保存当前进度...")
                break
            except Exception as e:
                logger.error(f"处理问题时发生未预期错误: {e}")
                continue

        # 完成处理
        total_time = time.time() - start_time
        self.stats["end_time"] = datetime.now().isoformat()
        self.stats["total_processing_time"] = total_time

        # 保存最终统计
        self._save_final_stats()

        # 打印最终报告
        self._print_final_report()

    def _print_progress_stats(self):
        """打印进度统计"""
        logger.info("=" * 50)
        logger.info("进度统计:")
        logger.info(f"已处理问题: {self.stats['processed_issues']}")

        for strategy, stats in self.stats["strategy_stats"].items():
            total = stats["success"] + stats["failed"]
            success_rate = (stats["success"] / total * 100) if total > 0 else 0
            logger.info(
                f"{strategy}: 成功 {stats['success']}, 失败 {stats['failed']} (成功率: {success_rate:.1f}%)"
            )

        logger.info("=" * 50)

    def _save_final_stats(self):
        """保存最终统计信息"""
        stats_file = self.output_dir / "evaluation_statistics.json"
        try:
            with open(stats_file, "w", encoding="utf-8") as f:
                json.dump(self.stats, f, ensure_ascii=False, indent=2)
            logger.info(f"统计信息已保存到: {stats_file}")
        except Exception as e:
            logger.error(f"保存统计信息失败: {e}")

    def _print_final_report(self):
        """打印最终报告"""
        logger.info("\n" + "=" * 60)
        logger.info("🎉 评估完成!")
        logger.info("=" * 60)
        logger.info(f"总问题数: {self.stats['total_issues']}")
        logger.info(f"已处理问题: {self.stats['processed_issues']}")
        logger.info(f"处理时间: {self.stats.get('total_processing_time', 0):.2f} 秒")

        logger.info("\n各策略统计:")
        for strategy, stats in self.stats["strategy_stats"].items():
            total = stats["success"] + stats["failed"]
            success_rate = (stats["success"] / total * 100) if total > 0 else 0
            logger.info(
                f"  {strategy:>10}: 成功 {stats['success']}, 失败 {stats['failed']} (成功率: {success_rate:.1f}%)"
            )

        if self.stats["processing_errors"]:
            logger.info(f"\n失败问题数: {len(self.stats['processing_errors'])}")

        logger.info(f"\n📁 结果文件位置:")
        for strategy in ["basic", "enhanced", "compressed"]:
            strategy_dir = self.output_dir / strategy
            if strategy_dir.exists():
                json_files = list(strategy_dir.glob("*/*.json"))
                logger.info(f"  {strategy}: {len(json_files)} 个文件")

        logger.info(f"\n📊 统计文件: {self.output_dir}/evaluation_statistics.json")
        logger.info(f"📝 日志文件: rag_run.log")

    def run_single_question(self, question_id: str) -> Dict:
        """
        运行单个问题的RAG评估

        Args:
            question_id: 问题ID

        Returns:
            包含评估结果的字典
        """
        logger.info(f"开始处理单个问题: {question_id}")

        # 查找对应的问题
        target_issue = None
        for issue in self.issues:
            if str(issue.get("id", "")) == str(question_id):
                target_issue = issue
                break

        if not target_issue:
            logger.error(f"未找到ID为 {question_id} 的问题")
            return {"error": f"Question ID {question_id} not found", "status": "failed"}

        logger.info(f"找到问题: {target_issue.get('issue_title', '')[:50]}...")

        # 处理该问题
        start_time = time.time()
        results = self.process_single_issue(target_issue)
        processing_time = time.time() - start_time

        # 构造返回结果
        result_summary = {
            "question_id": question_id,
            "issue_title": target_issue.get("issue_title", ""),
            "repo": target_issue.get("repo", ""),
            "processing_time": processing_time,
            "status": "completed",
            "strategies_processed": list(results.keys()),
            "results": results,
            "timestamp": datetime.now().isoformat(),
        }

        # 统计成功失败情况
        success_count = 0
        total_strategies = 3  # basic, enhanced, compressed

        for strategy in ["basic", "enhanced", "compressed"]:
            output_file = self.output_dir / strategy / f"{question_id}.json"
            if output_file.exists():
                success_count += 1

        result_summary["success_rate"] = success_count / total_strategies

        logger.info(f"问题 {question_id} 处理完成")
        logger.info(f"处理时间: {processing_time:.2f}秒")
        logger.info(f"成功策略: {success_count}/{total_strategies}")

        return result_summary


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="RAG评估运行器")
    parser.add_argument("--start", type=int, default=0, help="开始处理的问题索引")
    parser.add_argument("--max", type=int, help="最大处理问题数")
    parser.add_argument("--api-key", help="DeepSeek API密钥")
    parser.add_argument("--question-id", help="运行单个问题的ID")

    args = parser.parse_args()

    try:
        runner = RAGEvaluationRunner(api_key=args.api_key)

        if args.question_id:
            # 处理单个问题
            result = runner.run_single_question(args.question_id)
            if result.get("status") == "failed":
                logger.error(f"处理问题 {args.question_id} 失败")
                sys.exit(1)
            else:
                logger.info("单个问题处理完成!")
                print("\n结果摘要:")
                print(f"问题ID: {result['question_id']}")
                print(f"处理时间: {result['processing_time']:.2f}秒")
                print(f"成功率: {result['success_rate']:.1%}")
                print(f"成功策略: {result['strategies_processed']}")
        else:
            # 批量处理
            runner.run_all(start_index=args.start, max_issues=args.max)

    except Exception as e:
        logger.error(f"运行失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
