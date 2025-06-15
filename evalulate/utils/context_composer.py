import json
import sys
from pathlib import Path

current_dir = Path(__file__).parent
configs_dir = current_dir.parent / "configs"
sys.path.insert(0, str(configs_dir))

from prompts import system_prompt, format_user_prompt_from_rag


def compose_chat_with_rag(
    dp: dict,
    rag_json_path: str,
    strategy: str = "basic",
):
    # 读取RAG内容块
    with open(rag_json_path, "r", encoding="utf-8") as f:
        rag_data = json.load(f)

    # 获取对应问题和策略下的内容块
    rag_chunks = rag_data.get(strategy, [])

    # 组装user prompt
    repo_name = f"{dp['repo']}"
    issue_text = f"{dp['issue_title']}\n{dp['issue_body']}"

    user_prompt = format_user_prompt_from_rag(repo_name, issue_text, rag_chunks)

    # 构造消息
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    return messages
