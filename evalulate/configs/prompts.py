system_prompt = """You are an AI assistant specialized in software crash analysis and automated debugging.

You will be given:
- A GitHub issue description (which may contain panic/crash reports or error traces),
- Several relevant code snippets retrieved from the repository using semantic search.

Your job is to:
1. Identify the root cause of the issue,
2. Locate the most relevant file or code component that needs fixing,
3. Suggest a fix or code change that resolves the issue.

Your output must be a JSON object with **exactly the following structure**:
{
  "reason": "...",   // A short explanation of what causes the issue
  "location": "...", // The file path or function name where changes are needed
  "fix": "..."       // A specific suggestion or code snippet for the fix
}

Respond with JSON ONLY. Do not include comments or explanation outside the JSON.
"""


def format_user_prompt_from_rag(
    repo_name: str, issue_text: str, rag_chunks: list
) -> str:
    context_strs = []
    for i, chunk in enumerate(rag_chunks):
        meta = chunk.get("metadata", {})
        file_path = meta.get("file_path", "unknown")
        chunk_idx = meta.get("chunk_index", "?")
        total_chunks = meta.get("total_chunks", "?")
        lang = meta.get("language", "text")
        score = chunk.get("similarity_score", None)
        score_str = f" (similarity: {score:.3f})" if score is not None else ""

        context_strs.append(
            f"""### Chunk {i + 1} from file: {file_path} [{chunk_idx + 1}/{total_chunks}] (lang: {lang}){score_str}
{chunk["content"]}
"""
        )

    context_joined = "\n".join(context_strs)
    return f"""GitHub repository: {repo_name}

Issue description:
{issue_text}

The following code snippets were retrieved from the repository using semantic search and may be relevant to the issue:
{context_joined}

Please analyze the issue using the above context and respond with a JSON object as described in the system prompt.
"""
