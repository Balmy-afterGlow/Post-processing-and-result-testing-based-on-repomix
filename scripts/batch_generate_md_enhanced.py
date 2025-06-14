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
    git_enhanced_metadata: dict | None = None,
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

    # 构建文件信息
    files_info = {"standard": standard_metadata, "compressed": compressed_metadata}

    total_files = 2
    total_size = standard_metadata.get("file_size_bytes", 0) + compressed_metadata.get(
        "file_size_bytes", 0
    )
    total_tokens = standard_tokens + compressed_tokens

    # 如果有Git增强版本，添加到元数据中
    if git_enhanced_metadata:
        files_info["git_enhanced"] = git_enhanced_metadata
        total_files = 3
        total_size += git_enhanced_metadata.get("file_size_bytes", 0)
        total_tokens += git_enhanced_metadata.get("token_counts", {}).get("gpt-4", 0)

    metadata = {
        "repository": {
            "name": repo,
            "sha": sha,
            "generation_date": datetime.now().isoformat(),
        },
        "files": files_info,
        "summary": {
            "total_files": total_files,
            "total_size_bytes": total_size,
            "total_tokens_gpt4": total_tokens,
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
    if git_enhanced_metadata:
        git_tokens = git_enhanced_metadata.get("token_counts", {}).get("gpt-4", 0)
        print(f"    Git-enhanced tokens (GPT-4): {git_tokens:,}")
    print(f"    Compression ratio: {token_ratio:.3f}")


def generate_md(repo: str, sha: str, env_name: str = "repomind"):
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

    # 生成带Git提交历史的版本
    git_enhanced_path = output_dir / f"repomix-output-{name}-with-git.md"
    print(f"[+] Generating Git-enhanced version: {git_enhanced_path}")
    git_enhanced_success = False
    try:
        # 调用 add_commit_info.py 脚本
        add_commit_script = Path(__file__).parent / "add_commit_info.py"
        subprocess.run(
            [
                "conda activate",
                f"{env_name}",
                "&&",
                "python",
                str(add_commit_script),
                str(standard_path),
                "-r",
                repo,  # 使用远程仓库
                "-o",
                str(git_enhanced_path),
                "-c",
                "5",  # 显示5条提交历史，避免文件过大
            ],
            check=True,
        )
        print("[+] Git-enhanced version created successfully")
        git_enhanced_success = True
    except subprocess.CalledProcessError as e:
        print(f"[!] Failed to create Git-enhanced version: {e}")

    # 计算元数据
    print(f"[+] Calculating metadata for {repo}...")
    standard_metadata = get_file_metadata(standard_path)
    compressed_metadata = get_file_metadata(compressed_path)

    # 如果成功生成了Git增强版本，也计算其元数据
    git_enhanced_metadata = None
    if git_enhanced_success and git_enhanced_path.exists():
        git_enhanced_metadata = get_file_metadata(git_enhanced_path)

    # 保存元数据
    save_metadata(
        output_dir,
        repo,
        sha,
        standard_metadata,
        compressed_metadata,
        git_enhanced_metadata,
    )

    return standard_metadata, compressed_metadata, git_enhanced_metadata


def main():
    repo_map = load_repos_from_issues()

    print(f"[+] Starting batch generation for {len(repo_map)} repositories...")

    for repo, sha in repo_map.items():
        try:
            print(f"\n[+] Processing {repo} (SHA: {sha})")
            standard_metadata, compressed_metadata, git_enhanced_metadata = generate_md(
                repo, sha
            )

        except subprocess.CalledProcessError as e:
            print(f"[!] Failed to generate for {repo}: {e}")
        except Exception as e:
            print(f"[!] Unexpected error for {repo}: {e}")


if __name__ == "__main__":
    main()
