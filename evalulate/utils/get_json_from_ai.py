#!/usr/bin/env python3
"""
从AI输出中安全提取JSON的工具函数

这个模块提供了从AI模型输出中安全提取特定格式JSON的功能，
能够处理AI输出中可能包含的额外文本、markdown代码块等情况。
"""

import json
import re
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


def extract_json_from_ai_response(
    ai_response: str,
    required_keys: List[str] = ["reason", "location", "fix"],
    default_values: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    从AI响应中安全地提取JSON数据

    Args:
        ai_response: AI模型的原始响应文本
        required_keys: 必需的JSON键列表
        default_values: 当提取失败时使用的默认值字典

    Returns:
        提取到的JSON字典，如果失败则返回默认值或空字典
    """
    if default_values is None:
        default_values = {
            "reason": "Failed to extract reason from AI response",
            "location": [],
            "fix": "Failed to extract fix from AI response",
        }

    # 清理响应文本
    cleaned_response = clean_ai_response(ai_response)

    # 尝试多种方法提取JSON
    json_data = None

    # 方法1: 直接解析整个响应
    json_data = try_parse_json(cleaned_response)

    # 方法2: 查找JSON代码块
    if json_data is None:
        json_data = extract_json_from_code_blocks(cleaned_response)

    # 方法3: 查找大括号包围的JSON
    if json_data is None:
        json_data = extract_json_from_braces(cleaned_response)

    # 方法4: 使用正则表达式提取字段
    if json_data is None:
        json_data = extract_fields_with_regex(cleaned_response)

    # 验证和修复提取的JSON
    if json_data is not None:
        json_data = validate_and_fix_json(json_data, required_keys, default_values)
    else:
        logger.warning("All JSON extraction methods failed, using default values")
        json_data = default_values.copy()

    return json_data


def clean_ai_response(response: str) -> str:
    """清理AI响应文本"""
    # 移除多余的空白字符
    response = response.strip()

    # 移除可能的markdown标记
    response = re.sub(r"^```.*?\n", "", response, flags=re.MULTILINE)
    response = re.sub(r"\n```$", "", response, flags=re.MULTILINE)

    # 移除可能的解释文本（在JSON前后）
    # 查找第一个 { 和最后一个 }
    first_brace = response.find("{")
    last_brace = response.rfind("}")

    if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
        response = response[first_brace : last_brace + 1]

    return response


def try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    """尝试直接解析JSON"""
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, ValueError) as e:
        logger.debug(f"Direct JSON parsing failed: {e}")
    return None


def extract_json_from_code_blocks(text: str) -> Optional[Dict[str, Any]]:
    """从markdown代码块中提取JSON"""
    # 匹配 ```json 或 ``` 包围的代码块
    patterns = [
        r"```json\s*(.*?)\s*```",
        r"```\s*(.*?)\s*```",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            json_data = try_parse_json(match.strip())
            if json_data is not None:
                return json_data

    return None


def extract_json_from_braces(text: str) -> Optional[Dict[str, Any]]:
    """从大括号包围的文本中提取JSON"""
    # 查找最外层的大括号对
    brace_count = 0
    start_idx = -1

    for i, char in enumerate(text):
        if char == "{":
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == "}":
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                json_text = text[start_idx : i + 1]
                json_data = try_parse_json(json_text)
                if json_data is not None:
                    return json_data

    return None


def extract_fields_with_regex(text: str) -> Optional[Dict[str, Any]]:
    """使用正则表达式提取特定字段"""
    result = {}

    # 提取reason字段
    reason_patterns = [
        r'"reason"\s*:\s*"([^"]*)"',
        r'"reason"\s*:\s*\'([^\']*)\'',
        r'reason\s*:\s*"([^"]*)"',
        r"reason\s*:\s*\'([^\']*)\'",
    ]

    for pattern in reason_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            result["reason"] = match.group(1).strip()
            break

    # 提取location字段（数组格式）
    location_patterns = [
        r'"location"\s*:\s*\[([^\]]*)\]',
        r"location\s*:\s*\[([^\]]*)\]",
    ]

    for pattern in location_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            location_str = match.group(1).strip()
            # 解析数组内容
            locations = []
            for item in re.findall(r'"([^"]*)"', location_str):
                locations.append(item.strip())
            if not locations:
                # 尝试单引号
                for item in re.findall(r"'([^']*)'", location_str):
                    locations.append(item.strip())
            result["location"] = locations
            break

    # 提取fix字段
    fix_patterns = [
        r'"fix"\s*:\s*"([^"]*)"',
        r'"fix"\s*:\s*\'([^\']*)\'',
        r'fix\s*:\s*"([^"]*)"',
        r"fix\s*:\s*\'([^\']*)\'",
    ]

    for pattern in fix_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            result["fix"] = match.group(1).strip()
            break

    # 如果提取到了至少一个字段，返回结果
    if result:
        return result

    return None


def validate_and_fix_json(
    json_data: Dict[str, Any], required_keys: List[str], default_values: Dict[str, Any]
) -> Dict[str, Any]:
    """验证和修复JSON数据"""
    result: Dict[str, Any] = {}

    for key in required_keys:
        if key in json_data and json_data[key] is not None:
            value = json_data[key]

            # 特殊处理location字段，确保它是列表
            if key == "location":
                if isinstance(value, str):
                    # 如果是字符串，转换为列表
                    result[key] = [value] if value.strip() else []
                elif isinstance(value, list):
                    result[key] = value
                else:
                    result[key] = []
            else:
                # 确保其他字段是字符串
                if isinstance(value, (str, int, float)):
                    result[key] = str(value)
                else:
                    result[key] = default_values.get(key, "")
        else:
            # 使用默认值
            result[key] = default_values.get(key, "")

    return result


def test_json_extraction():
    """测试JSON提取功能"""
    test_cases = [
        # 标准JSON
        '{"reason": "Test reason", "location": ["file1.py", "file2.py"], "fix": "Test fix"}',
        # 带markdown代码块
        '```json\n{"reason": "Test reason", "location": ["file1.py"], "fix": "Test fix"}\n```',
        # 带解释文本
        'Here is the analysis:\n{"reason": "Test reason", "location": ["file1.py"], "fix": "Test fix"}\nThis should work.',
        # 格式不完整
        'reason: "Test reason"\nlocation: ["file1.py"]\nfix: "Test fix"',
        # 完全无效
        "This is not JSON at all",
    ]

    for i, test_case in enumerate(test_cases):
        print(f"\nTest case {i + 1}:")
        print(f"Input: {test_case}")
        result = extract_json_from_ai_response(test_case)
        print(f"Output: {result}")


if __name__ == "__main__":
    test_json_extraction()
