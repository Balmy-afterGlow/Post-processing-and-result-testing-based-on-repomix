import json
from pathlib import Path
import subprocess


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


def main():
    repo_map = load_repos_from_issues()
    for repo, sha in repo_map.items():
        try:
            generate_md(repo, sha)
        except subprocess.CalledProcessError as e:
            print(f"[!] Failed to generate for {repo}: {e}")


if __name__ == "__main__":
    main()
