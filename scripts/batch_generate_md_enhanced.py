import json
from pathlib import Path
import subprocess
from datetime import datetime
import tiktoken


def load_repos_from_issues(json_path="../evalulate/datasets/issues.json"):
    with open(json_path, "r", encoding="utf-8") as f:
        issues = json.load(f)

    repo_sha = {}
    for item in issues:
        repo = item.get("repo")
        sha = item.get("base_sha")
        if repo and sha and repo not in repo_sha:
            repo_sha[repo] = sha
    return repo_sha


# 常用的忽略文件（你可以按需要微调）
ignore_patterns = "*.md,*.MD,*.ipynb,docs/*,test/*,tests/*,examples/*"


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """
    计算文本的 token 数量

    Args:
        text: 要计算的文本
        model: 模型名称，默认为 gpt-4

    Returns:
        int: token 数量
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception as e:
        print(f"[!] Token calculation failed for {model}: {e}")
        # 如果 tiktoken 失败，使用简单的估算方法（约4个字符=1个token）
        return len(text) // 4


def get_file_metadata(file_path: Path) -> dict:
    """
    获取文件的元数据

    Args:
        file_path: 文件路径

    Returns:
        dict: 包含文件元数据的字典
    """
    if not file_path.exists():
        return {}

    # 读取文件内容
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"[!] Failed to read file {file_path}: {e}")
        return {}

    # 获取文件信息
    stat = file_path.stat()

    # 计算 tokens（多个模型）
    token_gpt4 = count_tokens(content, "gpt-4")
    token_estimated = len(content) // 4

    return {
        "file_name": file_path.name,
        "file_size_mb": round(stat.st_size / (1024 * 1024), 2),
        "character_count": len(content),
        "line_count": content.count("\n") + 1,
        "word_count": len(content.split()),
        "token_counts": {
            "gpt-4": token_gpt4,
            "estimated": token_estimated,
        },
        "created_time": datetime.fromtimestamp(stat.st_ctime).isoformat(),
        "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "generation_time": datetime.now().isoformat(),
    }


def save_metadata(
    output_dir: Path,
    repo: str,
    sha: str,
    standard_metadata: dict,
    compressed_metadata: dict,
):
    """
    保存元数据到 JSON 文件
    """
    # 计算压缩比率
    size_ratio = 0
    token_ratio = 0

    if standard_metadata.get("file_size_bytes", 0) > 0:
        size_ratio = (
            compressed_metadata.get("file_size_bytes", 0)
            / standard_metadata["file_size_bytes"]
        )

    standard_tokens = standard_metadata.get("token_counts", {}).get("gpt-4", 0)
    compressed_tokens = compressed_metadata.get("token_counts", {}).get("gpt-4", 0)
    if standard_tokens > 0:
        token_ratio = compressed_tokens / standard_tokens

    metadata = {
        "repository": {
            "name": repo,
            "sha": sha,
            "generation_date": datetime.now().isoformat(),
        },
        "files": {"standard": standard_metadata, "compressed": compressed_metadata},
        "summary": {
            "total_files": 2,
            "total_size_bytes": standard_metadata.get("file_size_bytes", 0)
            + compressed_metadata.get("file_size_bytes", 0),
            "total_tokens_gpt4": standard_tokens + compressed_tokens,
            "compression_ratio": {
                "size": round(size_ratio, 3),
                "tokens": round(token_ratio, 3),
            },
        },
    }

    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"[+] Metadata saved: {metadata_file}")
    print(f"    Standard tokens (GPT-4): {standard_tokens:,}")
    print(f"    Compressed tokens (GPT-4): {compressed_tokens:,}")
    print(f"    Compression ratio: {token_ratio:.3f}")


def generate_md(repo: str, sha: str):
    owner, name = repo.split("/")
    output_dir = Path(f"../repomix_md/repository-{name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    base_args = [
        "repomix",
        "--remote",
        repo,
        "--remote-branch",
        sha,
        "--style",
        "markdown",
        "--ignore",
        ignore_patterns,
    ]

    # 标准md
    standard_path = output_dir / f"repomix-output-{name}.md"
    print(f"[+] Generating: {standard_path}")
    subprocess.run(base_args + ["-o", str(standard_path)], check=True)

    # 压缩md
    compressed_path = output_dir / f"repomix-output-{name}-compress.md"
    print(f"[+] Generating (compressed): {compressed_path}")
    subprocess.run(base_args + ["--compress", "-o", str(compressed_path)], check=True)

    # 计算元数据
    print(f"[+] Calculating metadata for {repo}...")
    standard_metadata = get_file_metadata(standard_path)
    compressed_metadata = get_file_metadata(compressed_path)

    # 保存元数据
    save_metadata(output_dir, repo, sha, standard_metadata, compressed_metadata)

    return standard_metadata, compressed_metadata


def main():
    repo_map = load_repos_from_issues()
    all_metadata = []
    success_count = 0
    failed_repos = []

    print(f"[+] Starting batch generation for {len(repo_map)} repositories...")

    for repo, sha in repo_map.items():
        try:
            print(f"\n[+] Processing {repo} (SHA: {sha})")
            standard_metadata, compressed_metadata = generate_md(repo, sha)

            # 收集统计信息
            all_metadata.append(
                {
                    "repo": repo,
                    "sha": sha,
                    "standard": standard_metadata,
                    "compressed": compressed_metadata,
                }
            )
            success_count += 1

        except subprocess.CalledProcessError as e:
            print(f"[!] Failed to generate for {repo}: {e}")
            failed_repos.append({"repo": repo, "error": str(e)})
        except Exception as e:
            print(f"[!] Unexpected error for {repo}: {e}")
            failed_repos.append({"repo": repo, "error": str(e)})


if __name__ == "__main__":
    main()
