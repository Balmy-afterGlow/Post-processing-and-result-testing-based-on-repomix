#!/usr/bin/env python3
"""
RAG评估结果图表生成器
生成类似GAIA Benchmark风格的对比图表
"""

import os
import sys
import json
import argparse
import logging
from typing import Dict, List, Any, Optional
import numpy as np

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 第三方库导入
try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print(
        "Warning: matplotlib and/or seaborn not available. Please install them for chart generation."
    )

# 设置matplotlib中文字体支持
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


class RAGChartGenerator:
    """RAG评估结果图表生成器"""

    def __init__(
        self,
        results_dir: str = "../results/based_on_rag",
        output_dir: str = "../results/charts/based_on_rag",
    ):
        """
        初始化图表生成器

        Args:
            results_dir: 评估结果目录
            output_dir: 图表输出目录
        """
        self.results_dir = os.path.abspath(results_dir)
        self.output_dir = os.path.abspath(output_dir)
        # 调整策略顺序：compressed在上，enhanced在中，basic在下
        self.strategies = ["compressed", "enhanced", "basic"]

        # 设置日志
        self.setup_logging()

        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)

        # 定义要展示的关键指标
        self.key_metrics = {
            "text_similarity": {
                "display_name": "Text Similarity",
                "path": ["text_similarity", "average_text_similarity"],
                "description": "Average semantic similarity between AI output and gold standard",
            },
            "location_matching": {
                "display_name": "Location Matching",
                "path": ["location_matching", "f1"],
                "description": "F1 score for code location identification accuracy",
            },
            "ranking_quality": {
                "display_name": "Ranking Quality",
                "path": ["ranking_metrics", "map_score"],
                "description": "Mean Average Precision for location ranking",
            },
            "structure_completeness": {
                "display_name": "Structure Completeness",
                "path": ["structure_completeness", "overall_completeness"],
                "description": "Completeness of output structure and content",
            },
            "overall_performance": {
                "display_name": "Overall Performance",
                "path": ["overall_scores", "weighted_score"],
                "description": "Weighted combination of all evaluation metrics",
            },
        }

        # 设置GAIA风格的黑、浅灰、白配色方案
        self.colors = {
            "compressed": "#171717",  # 深黑色 (最高级策略)
            "enhanced": "#5B5B5B",  # 浅灰色 (中级策略)
            "basic": "#B8B8B8",  # 白色 (基础策略)
        }

        # 对应的文字颜色 - 根据条色自动调整
        self.text_colors = {
            "compressed": "#FFFFFF",  # 黑条用白字
            "enhanced": "#FFFFFF",  # 灰条用白字
            "basic": "#2C2C2C",  # 白条用深字
        }

        # 设置图表样式
        self.setup_style()

    def setup_logging(self):
        """设置日志配置"""
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)

    def setup_style(self):
        """设置图表样式"""
        if not PLOTTING_AVAILABLE:
            return

        # 设置seaborn样式和更美观的配置
        sns.set_style("whitegrid")
        plt.rcParams.update(
            {
                "figure.figsize": (14, 9),  # 增大画布尺寸
                "font.size": 12,
                "axes.titlesize": 18,
                "axes.labelsize": 14,
                "xtick.labelsize": 11,
                "ytick.labelsize": 11,
                "legend.fontsize": 12,
                "figure.titlesize": 20,
                "font.family": "sans-serif",
                "font.weight": "normal",
            }
        )

    def load_evaluation_result(
        self, strategy: str, question_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        加载评估结果

        Args:
            strategy: 策略名称
            question_id: 问题ID

        Returns:
            评估结果字典，失败返回None
        """
        eval_file = os.path.join(
            self.results_dir, strategy, question_id, f"{question_id}_evaluation.json"
        )

        try:
            with open(eval_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.logger.warning(f"Failed to load {eval_file}: {e}")
            return None

    def extract_metric_value(
        self, data: Dict[str, Any], metric_path: List[str]
    ) -> float:
        """
        从嵌套字典中提取指标值

        Args:
            data: 数据字典
            metric_path: 指标路径列表

        Returns:
            指标值，失败返回0.0
        """
        try:
            current = data["metrics"]
            for key in metric_path:
                current = current[key]
            return float(current)
        except (KeyError, TypeError, ValueError):
            return 0.0

    def collect_question_data(
        self, question_id: str
    ) -> Optional[Dict[str, Dict[str, float]]]:
        """
        收集单个问题的所有策略数据

        Args:
            question_id: 问题ID

        Returns:
            包含所有策略和指标数据的字典
        """
        question_data = {}

        for strategy in self.strategies:
            eval_result = self.load_evaluation_result(strategy, question_id)
            if eval_result is None:
                continue

            strategy_metrics = {}
            for metric_key, metric_config in self.key_metrics.items():
                value = self.extract_metric_value(
                    eval_result, list(metric_config["path"])
                )
                strategy_metrics[metric_key] = value

            question_data[strategy] = strategy_metrics

        # 检查是否有足够的数据
        if len(question_data) < 2:
            self.logger.warning(f"Insufficient data for question {question_id}")
            return None

        return question_data

    def create_gaia_style_chart(
        self, question_id: str, data: Dict[str, Dict[str, float]]
    ) -> str:
        """
        创建GAIA风格的对比图表

        Args:
            question_id: 问题ID
            data: 问题数据

        Returns:
            生成的图表文件路径
        """
        if not PLOTTING_AVAILABLE:
            self.logger.error("Plotting libraries not available")
            return ""

        # 创建图表 - 增大尺寸并设置高DPI
        fig, ax = plt.subplots(figsize=(14, 9), dpi=100)

        # 设置更优雅的背景
        fig.patch.set_facecolor("#E8E8E8")  # 灰色背景
        ax.set_facecolor("#E8E8E8")  # 灰色图表区域

        # 准备数据
        metrics = list(self.key_metrics.keys())
        # 确保策略按正确顺序排列：compressed在上，enhanced在中，basic在下
        strategies = [s for s in self.strategies if s in data]

        # 计算条形图的位置 - 增加条形高度使其更突出
        y_positions = np.arange(len(metrics))
        bar_height = 0.18  # 稍微减小以避免重叠

        # 为每个策略绘制条形图 - 确保compressed在上，enhanced在中，basic在下
        for i, strategy in enumerate(strategies):
            values = [data[strategy].get(metric, 0.0) for metric in metrics]

            # 调整y位置 - 使用策略在self.strategies中的索引来确定正确顺序
            strategy_index = self.strategies.index(strategy)
            y_pos = y_positions + (strategy_index - 1) * bar_height

            # 绘制条形图
            bars = ax.barh(
                y_pos,
                values,
                bar_height,
                label=f"{strategy.capitalize()}",
                color=self.colors.get(strategy, "#BDBDBD"),
                alpha=0.9,
                edgecolor="#FFFFFF",
                linewidth=1.0,
            )

            # 在条形右端添加数值标签 - 细字体，颜色根据条色自动调整
            for j, (bar, value) in enumerate(zip(bars, values)):
                if value > 0:
                    # 数值放在条形右端
                    text_x = bar.get_width() - 0.02
                    text_y = bar.get_y() + bar.get_height() / 2
                    text_color = self.text_colors[strategy]

                    ax.text(
                        text_x,
                        text_y,
                        f"{value:.1%}",
                        ha="right",
                        va="center",
                        fontweight="normal",  # 细字体
                        fontsize=10,
                        color=text_color,  # 统一使用深色字体，因为背景是白色
                    )

        # 设置坐标轴
        ax.set_yticks(y_positions)
        ax.set_yticklabels([str(self.key_metrics[m]["display_name"]) for m in metrics])
        ax.set_xlabel("Performance Score", fontsize=14, fontweight="bold")
        ax.set_xlim(0, 1.1)

        # 设置更柔和的网格
        ax.grid(
            True, axis="x", alpha=0.6, linestyle="-", linewidth=0.8, color="#999999"
        )
        ax.set_axisbelow(True)
        ax.grid(
            True, axis="y", alpha=0.15, linestyle=":", linewidth=0.4, color="#F0F0F0"
        )
        ax.set_axisbelow(True)

        # 设置柔和的标题
        title = f"RAG Evaluation Results\n{question_id.upper()}"
        ax.set_title(title, fontsize=20, fontweight="bold", pad=50, color="#424242")

        # 添加GAIA风格的横排圆形图例（放在标题下方）
        # 在标题下方创建图例
        legend_elements = []
        for strategy in self.strategies:  # 使用正确的顺序
            if strategy in data:
                color = self.colors.get(strategy, "#BDBDBD")
                legend_elements.append(
                    plt.Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        markerfacecolor=color,
                        markersize=12,
                        label=f"{strategy.capitalize()}",
                    )
                )

        ax.legend(
            handles=legend_elements,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.05),  # 调整图例位置更靠近标题
            ncol=3,  # 横排
            frameon=False,
            fontsize=12,
            columnspacing=2.5,
        )

        # 柔和的边框
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.2)
        ax.spines["left"].set_color("#666666")  # 更深的颜色
        ax.spines["bottom"].set_linewidth(1.2)
        ax.spines["bottom"].set_color("#666666")  # 更深的颜色

        # 添加更优雅的子标题说明
        subtitle = "Performance comparison across key evaluation metrics • Higher scores indicate better performance"
        fig.text(
            0.5,
            0.02,
            subtitle,
            ha="center",
            va="bottom",
            fontsize=11,
            style="italic",
            color="#666666",
            weight="normal",
        )

        # 调整布局并优化间距 - 为横排图例留出空间
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.12, top=0.85, left=0.15, right=0.95)

        # 保存高质量图表
        output_path = os.path.join(self.output_dir, f"{question_id}_comparison.png")
        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
            facecolor="#E8E8E8",  # 匹配背景色
            edgecolor="none",
            pad_inches=0.2,  # 增加边距
            format="png",
            pil_kwargs={"optimize": True},  # 优化文件大小
        )
        plt.close()

        self.logger.info(f"Generated chart: {output_path}")
        return output_path

    def create_summary_chart(
        self, all_data: Dict[str, Dict[str, Dict[str, float]]]
    ) -> str:
        """
        创建汇总对比图表

        Args:
            all_data: 所有问题的数据

        Returns:
            生成的汇总图表文件路径
        """
        if not PLOTTING_AVAILABLE:
            return ""

        # 计算每个策略每个指标的平均值
        strategy_averages: Dict[str, Dict[str, float]] = {}

        for strategy in self.strategies:
            strategy_averages[strategy] = {}
            for metric in self.key_metrics.keys():
                values = []
                for question_data in all_data.values():
                    if strategy in question_data and metric in question_data[strategy]:
                        values.append(question_data[strategy][metric])

                strategy_averages[strategy][metric] = (
                    float(np.mean(values)) if values else 0.0
                )

        # 创建雷达图
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))

        # 设置角度
        angles = np.linspace(0, 2 * np.pi, len(self.key_metrics), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))  # 闭合

        # 为每个策略绘制雷达图
        for strategy in self.strategies:
            if strategy not in strategy_averages:
                continue

            values = [
                strategy_averages[strategy][metric]
                for metric in self.key_metrics.keys()
            ]
            values += [values[0]]  # 闭合

            ax.plot(
                angles,
                values,
                "o-",
                linewidth=2.5,
                label=strategy.capitalize(),
                color=self.colors.get(strategy, "#7F8C8D"),
            )
            ax.fill(
                angles, values, alpha=0.3, color=self.colors.get(strategy, "#7F8C8D")
            )

        # 设置标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(
            [str(self.key_metrics[m]["display_name"]) for m in self.key_metrics.keys()]
        )

        # 设置范围
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"])
        ax.grid(True)

        # 添加标题和图例
        plt.title(
            "RAG Strategy Performance Summary\nAverage Scores Across All Questions",
            fontsize=16,
            fontweight="bold",
            pad=30,
        )
        plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))

        # 保存图表
        output_path = os.path.join(self.output_dir, "strategy_comparison_summary.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()

        self.logger.info(f"Generated summary chart: {output_path}")
        return output_path

    def generate_all_charts(self) -> Dict[str, List[str]]:
        """
        生成所有问题的图表

        Returns:
            生成的图表路径字典
        """
        if not PLOTTING_AVAILABLE:
            self.logger.error(
                "Cannot generate charts: plotting libraries not available"
            )
            return {}

        self.logger.info("Starting chart generation...")

        # 找到所有问题ID
        question_ids = set()
        for strategy in self.strategies:
            strategy_dir = os.path.join(self.results_dir, strategy)
            if os.path.exists(strategy_dir):
                for item in os.listdir(strategy_dir):
                    if os.path.isdir(os.path.join(strategy_dir, item)):
                        question_ids.add(item)

        if not question_ids:
            self.logger.warning("No question directories found")
            return {}

        self.logger.info(f"Found {len(question_ids)} questions to process")

        # 生成各个问题的图表
        generated_charts: Dict[str, List[str]] = {"individual": [], "summary": []}

        all_data = {}

        for question_id in sorted(question_ids):
            question_data = self.collect_question_data(question_id)
            if question_data:
                chart_path = self.create_gaia_style_chart(question_id, question_data)
                if chart_path:
                    generated_charts["individual"].append(chart_path)
                    all_data[question_id] = question_data

        # 生成汇总图表
        if all_data:
            summary_path = self.create_summary_chart(all_data)
            if summary_path:
                generated_charts["summary"].append(summary_path)

        self.logger.info(
            f"Chart generation completed. Generated {len(generated_charts['individual'])} individual charts and {len(generated_charts['summary'])} summary charts"
        )

        return generated_charts

    def generate_question_chart(self, question_id: str) -> Optional[str]:
        """
        生成单个问题的图表

        Args:
            question_id: 问题ID

        Returns:
            生成的图表路径，失败返回None
        """
        if not PLOTTING_AVAILABLE:
            self.logger.error("Cannot generate chart: plotting libraries not available")
            return None

        question_data = self.collect_question_data(question_id)
        if not question_data:
            self.logger.error(f"No data available for question {question_id}")
            return None

        return self.create_gaia_style_chart(question_id, question_data)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="RAG Evaluation Chart Generator")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="../results/based_on_rag",
        help="Directory containing evaluation results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../results/charts/based_on_rag",
        help="Directory to save generated charts",
    )
    parser.add_argument(
        "--question-id", type=str, help="Generate chart for specific question ID only"
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

    if not PLOTTING_AVAILABLE:
        print("Error: Required plotting libraries not available.")
        print("Please install them with: pip install matplotlib seaborn")
        return 1

    # 创建图表生成器
    generator = RAGChartGenerator(args.results_dir, args.output_dir)

    # 生成图表
    if args.question_id:
        # 生成单个问题的图表
        chart_path = generator.generate_question_chart(args.question_id)
        if chart_path:
            print(f"Generated chart for {args.question_id}: {chart_path}")
        else:
            print(f"Failed to generate chart for {args.question_id}")
            return 1
    else:
        # 生成所有图表
        charts = generator.generate_all_charts()

        print("\n" + "=" * 60)
        print("CHART GENERATION SUMMARY")
        print("=" * 60)
        print(f"Individual charts: {len(charts.get('individual', []))}")
        print(f"Summary charts: {len(charts.get('summary', []))}")
        print(f"Output directory: {args.output_dir}")

        if charts.get("individual"):
            print(f"\nSample individual chart: {charts['individual'][0]}")
        if charts.get("summary"):
            print(f"Summary chart: {charts['summary'][0]}")

    return 0


if __name__ == "__main__":
    exit(main())
