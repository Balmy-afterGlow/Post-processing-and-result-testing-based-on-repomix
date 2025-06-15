#!/usr/bin/env python3
"""
标准答案生成脚本

此脚本实现以下功能：
1. 读取 evalulate/datasets/issues.json 中的数据
2. 解析每个 issue 的 diff 内容，使用 DeepSeek-V3 生成 reason 和 fix
3. 使用 changed_files 作为 location
4. 按照 prompts 中定义的 JSON 格式生成标准答案
5. 保存到 results/gold_standard/ 目录，文件名为 {id}.json

环境配置：
    在项目根目录创建 .env 文件，添加：
    DEEPSEEK_API_KEY=your_deepseek_api_key_here

用法：
    python standard_generate.py [--output-dir OUTPUT_DIR] [--use-ai] [--api-key API_KEY]

选项：
    --output-dir: 指定输出目录（默认：../../results/gold_standard）
    --use-ai: 是否使用 DeepSeek AI 生成 reason 和 fix（默认：False，使用占位符）
    --api-key: DeepSeek API key（可选，会优先使用 .env 中的 DEEPSEEK_API_KEY）
"""

import json
import os
import argparse
import ast
from pathlib import Path
from typing import Dict, List, Any
import logging

# 导入JSON提取工具
from get_json_from_ai import extract_json_from_ai_response

# 设置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv

    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False
    logger.warning(
        "python-dotenv not installed. Use 'pip install python-dotenv' to load .env files."
    )

try:
    import openai

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    logger.warning(
        "OpenAI package not installed. Use 'pip install openai' to enable AI generation."
    )


class GoldStandardGenerator:
    """标准答案生成器"""

    def __init__(
        self, output_dir: str, use_ai: bool = False, api_key: str | None = None
    ):
        """
        初始化生成器

        Args:
            output_dir: 输出目录路径
            use_ai: 是否使用 AI 生成 reason 和 fix
            api_key: DeepSeek API key (如果为None则从环境变量或.env文件加载)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_ai = use_ai

        if use_ai:
            if not HAS_OPENAI:
                raise ImportError(
                    "OpenAI package not installed. Install with: pip install openai"
                )

            if not HAS_DOTENV:
                raise ImportError(
                    "python-dotenv package not installed. Install with: pip install python-dotenv"
                )

            # 从.env文件加载环境变量

            load_dotenv()

            # 获取API key
            if not api_key:
                api_key = os.getenv("DEEPSEEK_API_KEY")
                if not api_key:
                    raise ValueError(
                        "DEEPSEEK_API_KEY not found in environment variables or .env file"
                    )

            # 配置DeepSeek客户端
            self.client = openai.OpenAI(
                api_key=api_key, base_url="https://api.deepseek.com"
            )

    def load_issues(self, issues_file: str) -> List[Dict[str, Any]]:
        """加载 issues 数据"""
        try:
            with open(issues_file, "r", encoding="utf-8") as f:
                issues = json.load(f)
            logger.info(f"Loaded {len(issues)} issues from {issues_file}")
            return issues
        except Exception as e:
            logger.error(f"Failed to load issues file: {e}")
            raise

    def parse_changed_files(self, changed_files_str: str) -> List[str]:
        """解析 changed_files 字符串为列表"""
        try:
            # 尝试直接解析为 Python 列表
            return ast.literal_eval(changed_files_str)
        except (ValueError, SyntaxError):
            # 如果失败，尝试简单的字符串分割
            if changed_files_str.startswith("[") and changed_files_str.endswith("]"):
                # 移除方括号并按逗号分割
                files_str = changed_files_str[1:-1]
                files = [f.strip().strip("'\"") for f in files_str.split(",")]
                return [f for f in files if f]  # 过滤空字符串
            else:
                # 作为单个文件处理
                return [changed_files_str.strip()]

    def generate_ai_analysis(
        self, issue_title: str, issue_body: str, diff: str
    ) -> Dict[str, str]:
        """使用 AI 分析 issue 和 diff，生成 reason 和 fix"""
        system_prompt = """You are an AI assistant specialized in software bug analysis and code fixing.

You will be given:
- An issue title and description
- A git diff showing the actual fix that was applied

Your job is to:
1. Analyze the issue and understand what was wrong
2. Explain the root cause based on the diff
3. Describe the fix that was applied

Your output must be a JSON object with exactly the following structure:
{
  "reason": "A clear explanation of what caused the issue/bug",
  "fix": "A description of how the issue was fixed, based on the diff"
}

Respond with JSON ONLY. Do not include comments or explanation outside the JSON."""

        user_prompt = f"""Issue Title: {issue_title}

Issue Description:
{issue_body}

Git Diff (the actual fix that was applied):
{diff}

Please analyze the issue and the fix, then respond with a JSON object as described."""

        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",  # 使用DeepSeek-V3模型
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=800,
            )

            content = response.choices[0].message.content.strip()
            logger.debug(f"AI raw response: {content}")

            # 使用安全的JSON提取函数
            default_values = {
                "reason": "AI analysis of the issue",
                "fix": "AI analysis of the fix"
            }
            
            result = extract_json_from_ai_response(
                content, 
                required_keys=["reason", "fix"],
                default_values=default_values
            )
            
            return {
                "reason": result.get("reason", "AI analysis of the issue"),
                "fix": result.get("fix", "AI analysis of the fix"),
            }

        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            return {"reason": "AI analysis failed", "fix": "AI fix generation failed"}

    def generate_placeholder_analysis(self, issue_id: str) -> Dict[str, str]:
        """生成占位符的 reason 和 fix"""
        return {
            "reason": f"Root cause analysis for issue {issue_id} - to be analyzed",
            "fix": f"Fix implementation for issue {issue_id} - to be determined",
        }

    def generate_gold_standard_answer(self, issue: Dict[str, Any]) -> Dict[str, Any]:
        """为单个 issue 生成标准答案"""
        issue_id = issue.get("id", "unknown")

        # 解析 changed_files 作为 location
        changed_files_str = issue.get("changed_files", "[]")
        try:
            location = self.parse_changed_files(changed_files_str)
        except Exception as e:
            logger.warning(f"Failed to parse changed_files for {issue_id}: {e}")
            location = []

        # 生成 reason 和 fix
        if self.use_ai:
            analysis = self.generate_ai_analysis(
                issue.get("issue_title", ""),
                issue.get("issue_body", ""),
                issue.get("diff", ""),
            )
        else:
            analysis = self.generate_placeholder_analysis(issue_id)

        # 构建标准答案 JSON
        gold_standard = {
            "reason": analysis["reason"],
            "location": location,
            "fix": analysis["fix"],
        }

        return gold_standard

    def save_gold_standard(self, issue_id: str, gold_standard: Dict[str, Any]) -> str:
        """保存标准答案到文件"""
        output_file = self.output_dir / f"{issue_id}.json"

        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(gold_standard, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved gold standard for {issue_id} to {output_file}")
            return str(output_file)

        except Exception as e:
            logger.error(f"Failed to save gold standard for {issue_id}: {e}")
            raise

    def generate_all(self, issues_file: str) -> Dict[str, str]:
        """生成所有 issues 的标准答案"""
        issues = self.load_issues(issues_file)
        results = {}

        logger.info(f"Generating gold standard answers for {len(issues)} issues...")

        for i, issue in enumerate(issues, 1):
            issue_id = issue.get("id", f"issue_{i}")

            try:
                logger.info(f"Processing issue {i}/{len(issues)}: {issue_id}")

                # 生成标准答案
                gold_standard = self.generate_gold_standard_answer(issue)

                # 保存到文件
                output_file = self.save_gold_standard(issue_id, gold_standard)
                results[issue_id] = output_file

            except Exception as e:
                logger.error(f"Failed to process issue {issue_id}: {e}")
                continue

        logger.info(f"Successfully generated {len(results)} gold standard answers")
        return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Generate gold standard answers for issues"
    )
    parser.add_argument(
        "--output-dir",
        default="../results/gold_standard",
        help="Output directory for gold standard files (default: ../results/gold_standard)",
    )
    parser.add_argument(
        "--use-ai",
        action="store_true",
        help="Use DeepSeek AI to generate reason and fix (requires DEEPSEEK_API_KEY in .env or environment)",
    )
    parser.add_argument(
        "--api-key",
        help="DeepSeek API key (optional, will use DEEPSEEK_API_KEY from .env if not provided)",
    )
    parser.add_argument(
        "--issues-file",
        default="../datasets/issues.json",
        help="Path to issues.json file (default: ../datasets/issues.json)",
    )

    args = parser.parse_args()

    # 检查 issues 文件是否存在
    if not os.path.exists(args.issues_file):
        parser.error(f"Issues file not found: {args.issues_file}")

    try:
        # 创建生成器
        generator = GoldStandardGenerator(
            output_dir=args.output_dir, use_ai=args.use_ai, api_key=args.api_key
        )

        # 生成标准答案
        results = generator.generate_all(args.issues_file)

        print("\nGeneration completed!")
        print(f"Generated {len(results)} gold standard files in: {args.output_dir}")

        if args.use_ai:
            print("Used DeepSeek AI to generate reason and fix descriptions")
        else:
            print(
                "Used placeholder text for reason and fix (use --use-ai to enable DeepSeek AI generation)"
            )

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
