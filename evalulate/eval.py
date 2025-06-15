#!/usr/bin/env python3
"""
RAG评估系统 - 自动对AI输出与标准答案进行多维度评估
支持多种指标：F1、Top-K、MAP、结构完整性、置信度、AI分析正确性等
"""

import os
import sys
import json
import argparse
import logging
from dotenv import load_dotenv
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
import re

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 标准库导入
from difflib import SequenceMatcher
import numpy as np

# 第三方库导入（需要时安装）
try:
    import openai

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print(
        "Warning: openai package not available. AI similarity analysis will be disabled."
    )

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print(
        "Warning: scikit-learn not available. Advanced text similarity will use basic methods."
    )

# 本地模块导入
from configs.prompts import get_evaluation_prompt


load_dotenv()


class RAGEvaluator:
    """RAG评估器类"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        openai_base_url: Optional[str] = None,
    ):
        """
        初始化评估器

        Args:
            api_key: OpenAI API密钥，用于AI分析正确性评估
            openai_base_url: OpenAI API基础URL，支持自定义接口
        """
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.openai_base_url = openai_base_url or os.getenv(
            "OPENAI_BASE_URL", "https://api.deepseek.com"
        )

        # 设置日志
        self.setup_logging()

        # 初始化OpenAI客户端
        if OPENAI_AVAILABLE and self.api_key:
            try:
                self.client = openai.OpenAI(
                    api_key=self.api_key,
                    base_url=self.openai_base_url,
                    timeout=30.0,  # 设置超时时间
                )
                self.ai_analysis_enabled = True
                self.logger.info(
                    f"AI analysis enabled with base URL: {self.openai_base_url}"
                )
            except Exception as e:
                self.logger.error(f"Failed to initialize OpenAI client: {e}")
                self.ai_analysis_enabled = False
        else:
            self.ai_analysis_enabled = False
            if not OPENAI_AVAILABLE:
                self.logger.warning("OpenAI package not available")
            if not self.api_key:
                self.logger.warning("No API key provided")

        # 初始化TF-IDF向量化器
        if SKLEARN_AVAILABLE:
            self.vectorizer = TfidfVectorizer(
                stop_words="english", ngram_range=(1, 2), max_features=5000
            )

    def setup_logging(self):
        """设置日志配置"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler("evaluation.log"), logging.StreamHandler()],
        )
        self.logger = logging.getLogger(__name__)

    def load_json_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        加载JSON文件

        Args:
            file_path: JSON文件路径

        Returns:
            JSON数据字典，加载失败返回None
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.logger.error(f"Failed to load {file_path}: {e}")
            return None

    def save_json_file(self, data: Dict[str, Any], file_path: str) -> bool:
        """
        保存JSON文件

        Args:
            data: 要保存的数据
            file_path: 保存路径

        Returns:
            保存是否成功
        """
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            self.logger.error(f"Failed to save {file_path}: {e}")
            return False

    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        计算文本相似度

        Args:
            text1: 文本1
            text2: 文本2

        Returns:
            相似度分数 (0-1)
        """
        if not text1 or not text2:
            return 0.0

        # 使用多种方法计算相似度，取平均值
        similarities = []

        # 1. 字符级别的相似度
        char_similarity = SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
        similarities.append(char_similarity)

        # 2. 词级别的相似度
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        word_similarity = len(words1 & words2) / max(len(words1 | words2), 1)
        similarities.append(word_similarity)

        # 3. TF-IDF余弦相似度（如果可用）
        if SKLEARN_AVAILABLE:
            try:
                tfidf_matrix = self.vectorizer.fit_transform([text1, text2])
                cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][
                    0
                ]
                similarities.append(cosine_sim)
            except Exception:
                pass

        return np.mean(similarities)

    def calculate_list_similarity(
        self, list1: List[str], list2: List[str]
    ) -> Dict[str, float]:
        """
        计算列表相似度（用于location字段）

        Args:
            list1: 列表1
            list2: 列表2

        Returns:
            包含多种指标的字典
        """
        if not list1 or not list2:
            return {
                "jaccard": 0.0,
                "overlap": 0.0,
                "f1": 0.0,
                "precision": 0.0,
                "recall": 0.0,
            }

        set1 = set(list1)
        set2 = set(list2)

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        # Jaccard相似度
        jaccard = intersection / union if union > 0 else 0.0

        # 重叠系数
        overlap = (
            intersection / min(len(set1), len(set2))
            if min(len(set1), len(set2)) > 0
            else 0.0
        )

        # F1、精确率、召回率
        precision = intersection / len(set1) if len(set1) > 0 else 0.0
        recall = intersection / len(set2) if len(set2) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {
            "jaccard": jaccard,
            "overlap": overlap,
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }

    def calculate_top_k_accuracy(
        self, predicted: List[str], actual: List[str], k: int = 3
    ) -> float:
        """
        计算Top-K准确率

        Args:
            predicted: 预测列表
            actual: 实际列表
            k: 取前k个

        Returns:
            Top-K准确率
        """
        if not predicted or not actual:
            return 0.0

        predicted_k = predicted[:k]
        actual_set = set(actual)

        hits = sum(1 for item in predicted_k if item in actual_set)
        return hits / min(k, len(actual_set))

    def calculate_map_score(self, predicted: List[str], actual: List[str]) -> float:
        """
        计算Mean Average Precision (MAP)

        Args:
            predicted: 预测列表
            actual: 实际列表

        Returns:
            MAP分数
        """
        if not predicted or not actual:
            return 0.0

        actual_set = set(actual)
        if not actual_set:
            return 0.0

        score = 0.0
        num_hits = 0

        for i, item in enumerate(predicted):
            if item in actual_set:
                num_hits += 1
                precision_at_i = num_hits / (i + 1)
                score += precision_at_i

        return score / len(actual_set) if len(actual_set) > 0 else 0.0

    def calculate_structure_completeness(
        self, ai_output: Dict[str, Any], gold_standard: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        计算结构完整性

        Args:
            ai_output: AI输出
            gold_standard: 标准答案

        Returns:
            结构完整性指标
        """
        required_fields = ["reason", "location", "fix"]
        ai_fields = set(ai_output.keys())
        gold_fields = set(gold_standard.keys())

        # 字段覆盖率
        field_coverage = (
            len(ai_fields & gold_fields) / len(gold_fields) if gold_fields else 0.0
        )

        # 必需字段完整性
        required_completeness = len(ai_fields & set(required_fields)) / len(
            required_fields
        )

        # 字段内容完整性
        content_completeness = []
        for field in required_fields:
            if field in ai_output and field in gold_standard:
                ai_content = ai_output[field]
                gold_content = gold_standard[field]

                if isinstance(ai_content, str) and isinstance(gold_content, str):
                    if ai_content.strip() and gold_content.strip():
                        content_completeness.append(1.0)
                    else:
                        content_completeness.append(0.0)
                elif isinstance(ai_content, list) and isinstance(gold_content, list):
                    if ai_content and gold_content:
                        content_completeness.append(1.0)
                    else:
                        content_completeness.append(0.0)
                else:
                    content_completeness.append(0.5)  # 类型不匹配但存在
            else:
                content_completeness.append(0.0)

        avg_content_completeness = (
            np.mean(content_completeness) if content_completeness else 0.0
        )

        return {
            "field_coverage": field_coverage,
            "required_completeness": required_completeness,
            "content_completeness": avg_content_completeness,
            "overall_completeness": (
                field_coverage + required_completeness + avg_content_completeness
            )
            / 3,
        }

    def calculate_confidence_score(self, ai_output: Dict[str, Any]) -> float:
        """
        计算置信度分数（基于内容长度、详细程度等）

        Args:
            ai_output: AI输出

        Returns:
            置信度分数 (0-1)
        """
        confidence_factors = []

        # 1. 内容长度因子
        reason_length = len(ai_output.get("reason", ""))
        fix_length = len(ai_output.get("fix", ""))

        # 归一化长度分数（假设合理长度为100-1000字符）
        reason_score = min(reason_length / 500, 1.0) if reason_length > 0 else 0.0
        fix_score = min(fix_length / 500, 1.0) if fix_length > 0 else 0.0

        confidence_factors.extend([reason_score, fix_score])

        # 2. 位置信息完整性
        locations = ai_output.get("location", [])
        location_score = min(len(locations) / 3, 1.0) if locations else 0.0
        confidence_factors.append(location_score)

        # 3. 内容详细程度（基于句子数量）
        reason_sentences = len(re.split(r"[.!?]+", ai_output.get("reason", "")))
        fix_sentences = len(re.split(r"[.!?]+", ai_output.get("fix", "")))

        detail_score = min((reason_sentences + fix_sentences) / 6, 1.0)
        confidence_factors.append(detail_score)

        return np.mean(confidence_factors) if confidence_factors else 0.0

    async def calculate_ai_similarity(
        self, ai_output: Dict[str, Any], gold_standard: Dict[str, Any]
    ) -> Optional[float]:
        """
        使用AI模型计算语义相似性

        Args:
            ai_output: AI输出
            gold_standard: 标准答案

        Returns:
            AI评估的相似性分数 (0-1)，失败返回None
        """
        if not self.ai_analysis_enabled:
            return None

        try:
            # 构造评估prompt
            prompt = get_evaluation_prompt(ai_output, gold_standard)

            self.logger.debug(f"Sending request to {self.openai_base_url}")

            # 调用deepseek-V3或其他模型
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert code analysis evaluator.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=500,
                temperature=0.1,
                timeout=30.0,
            )

            result_text = response.choices[0].message.content.strip()
            self.logger.debug(f"AI response: {result_text}")

            # 从响应中提取分数
            score_match = re.search(r"score[:\s]*(\d+(?:\.\d+)?)", result_text.lower())
            if score_match:
                score = float(score_match.group(1))
                # 如果分数是0-100范围，转换为0-1
                if score > 1:
                    score = score / 100
                return min(max(score, 0.0), 1.0)

            # 尝试其他可能的分数格式
            score_patterns = [
                r"(\d+(?:\.\d+)?)\s*/\s*100",  # "85/100"
                r"(\d+(?:\.\d+)?)%",  # "85%"
                r"(\d+(?:\.\d+)?)\s*points",  # "85 points"
            ]

            for pattern in score_patterns:
                match = re.search(pattern, result_text.lower())
                if match:
                    score = float(match.group(1))
                    if score > 1:
                        score = score / 100
                    return min(max(score, 0.0), 1.0)

            self.logger.warning(
                f"Could not extract score from AI response: {result_text}"
            )
            return None

        except openai.APIConnectionError as e:
            self.logger.error(f"API connection error: {e}")
            self.logger.info("Please check your network connection and API base URL")
            return None
        except openai.APITimeoutError as e:
            self.logger.error(f"API timeout error: {e}")
            return None
        except openai.RateLimitError as e:
            self.logger.error(f"Rate limit exceeded: {e}")
            return None
        except openai.AuthenticationError as e:
            self.logger.error(f"Authentication error: {e}")
            self.logger.info("Please check your API key")
            return None
        except Exception as e:
            self.logger.error(f"AI similarity analysis failed: {e}")
            return None

    def evaluate_single_case(
        self, ai_output: Dict[str, Any], gold_standard: Dict[str, Any], question_id: str
    ) -> Dict[str, Any]:
        """
        评估单个案例

        Args:
            ai_output: AI输出
            gold_standard: 标准答案
            question_id: 问题ID

        Returns:
            评估结果
        """
        results = {
            "question_id": question_id,
            "timestamp": str(Path().resolve()),
            "metrics": {},
        }

        # 1. 文本相似度指标
        reason_similarity = self.calculate_text_similarity(
            ai_output.get("reason", ""), gold_standard.get("reason", "")
        )

        fix_similarity = self.calculate_text_similarity(
            ai_output.get("fix", ""), gold_standard.get("fix", "")
        )

        # 2. 位置匹配指标
        location_metrics = self.calculate_list_similarity(
            ai_output.get("location", []), gold_standard.get("location", [])
        )

        # 3. Top-K和MAP指标
        top_1_acc = self.calculate_top_k_accuracy(
            ai_output.get("location", []), gold_standard.get("location", []), k=1
        )

        top_3_acc = self.calculate_top_k_accuracy(
            ai_output.get("location", []), gold_standard.get("location", []), k=3
        )

        map_score = self.calculate_map_score(
            ai_output.get("location", []), gold_standard.get("location", [])
        )

        # 4. 结构完整性
        structure_metrics = self.calculate_structure_completeness(
            ai_output, gold_standard
        )

        # 5. 置信度分数
        confidence = self.calculate_confidence_score(ai_output)

        # 6. AI相似性分析（如果启用）
        ai_similarity = None
        if self.ai_analysis_enabled:
            try:
                import asyncio

                ai_similarity = asyncio.run(
                    self.calculate_ai_similarity(ai_output, gold_standard)
                )
            except Exception as e:
                self.logger.warning(
                    f"AI similarity analysis failed for {question_id}: {e}"
                )

        # 7. 综合F1分数
        overall_f1 = (
            reason_similarity * 0.4
            + fix_similarity * 0.4
            + location_metrics["f1"] * 0.2
        )

        # 组合所有指标
        results["metrics"] = {
            "text_similarity": {
                "reason_similarity": reason_similarity,
                "fix_similarity": fix_similarity,
                "average_text_similarity": (reason_similarity + fix_similarity) / 2,
            },
            "location_matching": location_metrics,
            "ranking_metrics": {
                "top_1_accuracy": top_1_acc,
                "top_3_accuracy": top_3_acc,
                "map_score": map_score,
            },
            "structure_completeness": structure_metrics,
            "confidence_score": confidence,
            "ai_similarity": {
                "enabled": self.ai_analysis_enabled,
                "score": ai_similarity,
                "status": "success"
                if ai_similarity is not None
                else "failed"
                if self.ai_analysis_enabled
                else "disabled",
            },
            "overall_scores": {
                "f1_score": overall_f1,
                "weighted_score": (
                    overall_f1 * 0.4
                    + structure_metrics["overall_completeness"] * 0.3
                    + confidence * 0.2
                    + (ai_similarity * 0.1 if ai_similarity is not None else 0.0)
                ),
            },
        }

        return results

    def evaluate_strategy(
        self, strategy_dir: str, gold_dir: str, strategy_name: str
    ) -> Dict[str, Any]:
        """
        评估单个策略下的所有案例

        Args:
            strategy_dir: 策略目录路径
            gold_dir: 标准答案目录路径
            strategy_name: 策略名称

        Returns:
            策略评估结果
        """
        self.logger.info(f"Evaluating strategy: {strategy_name}")

        strategy_results = {
            "strategy": strategy_name,
            "total_cases": 0,
            "successful_evaluations": 0,
            "failed_cases": [],
            "case_results": {},
            "summary_metrics": {},
        }

        # 遍历策略目录下的所有问题
        if not os.path.exists(strategy_dir):
            self.logger.error(f"Strategy directory not found: {strategy_dir}")
            return strategy_results

        question_dirs = [
            d
            for d in os.listdir(strategy_dir)
            if os.path.isdir(os.path.join(strategy_dir, d))
        ]

        strategy_results["total_cases"] = len(question_dirs)

        all_metrics = defaultdict(list)

        for question_id in question_dirs:
            try:
                # 读取AI输出
                ai_output_path = os.path.join(
                    strategy_dir, question_id, f"{question_id}.json"
                )
                ai_output = self.load_json_file(ai_output_path)

                if not ai_output:
                    strategy_results["failed_cases"].append(
                        {
                            "question_id": question_id,
                            "error": "Failed to load AI output",
                        }
                    )
                    continue

                # 读取标准答案
                gold_path = os.path.join(gold_dir, f"{question_id}.json")
                gold_standard = self.load_json_file(gold_path)

                if not gold_standard:
                    strategy_results["failed_cases"].append(
                        {
                            "question_id": question_id,
                            "error": "Failed to load gold standard",
                        }
                    )
                    continue

                # 执行评估
                evaluation_result = self.evaluate_single_case(
                    ai_output, gold_standard, question_id
                )

                # 保存评估结果到与AI输出同目录
                eval_output_path = os.path.join(
                    strategy_dir, question_id, f"{question_id}_evaluation.json"
                )
                if self.save_json_file(evaluation_result, eval_output_path):
                    self.logger.info(f"Saved evaluation result: {eval_output_path}")

                strategy_results["case_results"][question_id] = evaluation_result
                strategy_results["successful_evaluations"] += 1

                # 收集指标用于计算平均值
                metrics = evaluation_result["metrics"]
                all_metrics["reason_similarity"].append(
                    metrics["text_similarity"]["reason_similarity"]
                )
                all_metrics["fix_similarity"].append(
                    metrics["text_similarity"]["fix_similarity"]
                )
                all_metrics["location_f1"].append(metrics["location_matching"]["f1"])
                all_metrics["top_1_accuracy"].append(
                    metrics["ranking_metrics"]["top_1_accuracy"]
                )
                all_metrics["top_3_accuracy"].append(
                    metrics["ranking_metrics"]["top_3_accuracy"]
                )
                all_metrics["map_score"].append(metrics["ranking_metrics"]["map_score"])
                all_metrics["overall_completeness"].append(
                    metrics["structure_completeness"]["overall_completeness"]
                )
                all_metrics["confidence_score"].append(metrics["confidence_score"])
                all_metrics["f1_score"].append(metrics["overall_scores"]["f1_score"])
                all_metrics["weighted_score"].append(
                    metrics["overall_scores"]["weighted_score"]
                )

                # AI相似性指标（如果可用）
                if metrics["ai_similarity"]["score"] is not None:
                    all_metrics["ai_similarity_score"].append(
                        metrics["ai_similarity"]["score"]
                    )

            except Exception as e:
                self.logger.error(f"Error evaluating {question_id}: {e}")
                strategy_results["failed_cases"].append(
                    {"question_id": question_id, "error": str(e)}
                )

        # 计算汇总指标
        if all_metrics:
            strategy_results["summary_metrics"] = {
                metric: {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "median": np.median(values),
                }
                for metric, values in all_metrics.items()
            }

        self.logger.info(
            f"Strategy {strategy_name} evaluation completed. "
            f"Success: {strategy_results['successful_evaluations']}/{strategy_results['total_cases']}"
        )

        return strategy_results

    def run_evaluation(
        self, rag_results_dir: str, gold_standard_dir: str, strategies: List[str] = None
    ) -> Dict[str, Any]:
        """
        运行完整评估

        Args:
            rag_results_dir: RAG结果目录
            gold_standard_dir: 标准答案目录
            strategies: 要评估的策略列表，None表示评估所有策略

        Returns:
            完整评估结果
        """
        self.logger.info("Starting RAG evaluation system")

        if strategies is None:
            strategies = ["basic", "compressed", "enhanced"]

        evaluation_results = {
            "evaluation_timestamp": str(Path().resolve()),
            "rag_results_dir": rag_results_dir,
            "gold_standard_dir": gold_standard_dir,
            "strategies_evaluated": strategies,
            "strategy_results": {},
            "comparative_analysis": {},
        }

        # 评估每个策略
        for strategy in strategies:
            strategy_dir = os.path.join(rag_results_dir, strategy)
            if os.path.exists(strategy_dir):
                strategy_result = self.evaluate_strategy(
                    strategy_dir, gold_standard_dir, strategy
                )
                evaluation_results["strategy_results"][strategy] = strategy_result
            else:
                self.logger.warning(f"Strategy directory not found: {strategy_dir}")

        # 生成对比分析
        evaluation_results["comparative_analysis"] = self.generate_comparative_analysis(
            evaluation_results["strategy_results"]
        )

        # 保存完整评估结果
        overall_results_path = os.path.join(rag_results_dir, "evaluation_results.json")
        self.save_json_file(evaluation_results, overall_results_path)

        self.logger.info(
            f"Evaluation completed. Results saved to {overall_results_path}"
        )

        return evaluation_results

    def generate_comparative_analysis(
        self, strategy_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        生成策略对比分析

        Args:
            strategy_results: 各策略评估结果

        Returns:
            对比分析结果
        """
        if not strategy_results:
            return {}

        comparative_analysis = {
            "best_strategy_by_metric": {},
            "metric_rankings": {},
            "improvement_analysis": {},
        }

        # 主要指标列表
        key_metrics = [
            "f1_score",
            "weighted_score",
            "reason_similarity",
            "fix_similarity",
            "location_f1",
            "top_1_accuracy",
            "top_3_accuracy",
            "map_score",
            "overall_completeness",
            "confidence_score",
            "ai_similarity_score",
        ]

        # 为每个指标找出最佳策略
        for metric in key_metrics:
            best_strategy = None
            best_score = -1

            strategy_scores = {}

            for strategy, results in strategy_results.items():
                if (
                    "summary_metrics" in results
                    and metric in results["summary_metrics"]
                ):
                    score = results["summary_metrics"][metric]["mean"]
                    strategy_scores[strategy] = score

                    if score > best_score:
                        best_score = score
                        best_strategy = strategy

            if best_strategy:
                comparative_analysis["best_strategy_by_metric"][metric] = {
                    "strategy": best_strategy,
                    "score": best_score,
                    "all_scores": strategy_scores,
                }

        # 计算总体排名
        strategy_total_scores = defaultdict(float)
        strategy_metric_counts = defaultdict(int)

        for metric, metric_data in comparative_analysis[
            "best_strategy_by_metric"
        ].items():
            for strategy, score in metric_data["all_scores"].items():
                strategy_total_scores[strategy] += score
                strategy_metric_counts[strategy] += 1

        # 计算平均分数并排名
        strategy_avg_scores = {
            strategy: total_score / strategy_metric_counts[strategy]
            for strategy, total_score in strategy_total_scores.items()
        }

        comparative_analysis["metric_rankings"] = dict(
            sorted(strategy_avg_scores.items(), key=lambda x: x[1], reverse=True)
        )

        # 改进分析（相对于basic策略）
        if "basic" in strategy_avg_scores:
            basic_score = strategy_avg_scores["basic"]
            for strategy, score in strategy_avg_scores.items():
                if strategy != "basic":
                    improvement = (
                        ((score - basic_score) / basic_score) * 100
                        if basic_score > 0
                        else 0
                    )
                    comparative_analysis["improvement_analysis"][strategy] = {
                        "improvement_percentage": improvement,
                        "absolute_improvement": score - basic_score,
                    }

        return comparative_analysis

    def test_ai_connection(self) -> bool:
        """
        测试AI API连接

        Returns:
            连接是否成功
        """
        if not self.ai_analysis_enabled:
            self.logger.info("AI analysis is disabled")
            return False

        try:
            self.logger.info("Testing AI API connection...")

            # 发送简单的测试请求
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {
                        "role": "user",
                        "content": "Hello, please respond with 'Connection test successful'",
                    }
                ],
                max_tokens=50,
                temperature=0.1,
                timeout=15.0,
            )

            result = response.choices[0].message.content.strip()
            self.logger.info(f"AI API connection test successful. Response: {result}")
            return True

        except openai.APIConnectionError as e:
            self.logger.error(f"API connection test failed: {e}")
            self.logger.info("Possible solutions:")
            self.logger.info("1. Check your internet connection")
            self.logger.info("2. Verify the API base URL is correct")
            self.logger.info("3. Check if there are any firewall restrictions")
            return False
        except openai.AuthenticationError as e:
            self.logger.error(f"Authentication failed: {e}")
            self.logger.info("Please verify your API key is correct")
            return False
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="RAG Evaluation System")
    parser.add_argument(
        "--rag-dir",
        type=str,
        default="./results/based_on_rag",
        help="RAG results directory",
    )
    parser.add_argument(
        "--gold-dir",
        type=str,
        default="./results/gold_standard",
        help="Gold standard directory",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        nargs="+",
        default=["basic", "compressed", "enhanced"],
        help="Strategies to evaluate",
    )
    parser.add_argument(
        "--api-key", type=str, help="OpenAI API key for AI similarity analysis"
    )
    parser.add_argument("--openai-base-url", type=str, help="OpenAI API base URL")
    parser.add_argument(
        "--disable-ai",
        action="store_true",
        help="Disable AI similarity analysis (faster evaluation)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # 设置日志级别
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # 转换为绝对路径
    rag_dir = os.path.abspath(args.rag_dir)
    gold_dir = os.path.abspath(args.gold_dir)  # 创建评估器
    api_key = None if args.disable_ai else args.api_key
    evaluator = RAGEvaluator(api_key=api_key, openai_base_url=args.openai_base_url)

    # 测试AI连接（如果启用）
    if evaluator.ai_analysis_enabled and not args.disable_ai:
        if not evaluator.test_ai_connection():
            print(
                "\n⚠️  AI similarity analysis will be disabled due to connection issues."
            )
            print("The evaluation will continue with other metrics only.\n")
            evaluator.ai_analysis_enabled = False
    elif args.disable_ai:
        print("\n📝 AI similarity analysis disabled by user request.\n")

    # 运行评估
    results = evaluator.run_evaluation(rag_dir, gold_dir, args.strategies)

    # 打印汇总结果
    print("\n" + "=" * 60)
    print("RAG EVALUATION RESULTS SUMMARY")
    print("=" * 60)

    if (
        "comparative_analysis" in results
        and "metric_rankings" in results["comparative_analysis"]
    ):
        print("\nOverall Strategy Rankings:")
        for i, (strategy, score) in enumerate(
            results["comparative_analysis"]["metric_rankings"].items(), 1
        ):
            print(f"{i}. {strategy}: {score:.4f}")

    if (
        "comparative_analysis" in results
        and "improvement_analysis" in results["comparative_analysis"]
    ):
        print("\nImprovement over Basic Strategy:")
        for strategy, analysis in results["comparative_analysis"][
            "improvement_analysis"
        ].items():
            improvement = analysis["improvement_percentage"]
            print(f"{strategy}: {improvement:+.2f}%")

    print(
        f"\nDetailed results saved to: {os.path.join(rag_dir, 'evaluation_results.json')}"
    )
    print("Individual evaluation results saved alongside AI outputs.")


if __name__ == "__main__":
    main()
