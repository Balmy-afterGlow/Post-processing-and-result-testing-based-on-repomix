import os
import json
from typing import List, Dict
from dataclasses import dataclass
import re
import shutil

# 设置环境变量以使用本地缓存，避免网络请求
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1" 
os.environ["HF_DATASETS_OFFLINE"] = "1"

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import tiktoken


@dataclass
class GitCommitInfo:
    """Git提交信息结构"""

    commit_id: str
    author: str
    date: str
    message: str
    diff_content: str


@dataclass
class CodeBlock:
    """代码块结构"""

    file_path: str
    content: str
    language: str
    git_commits: List[GitCommitInfo]
    metadata: Dict


class MarkdownParser:
    """Markdown文档解析器"""

    def __init__(self):
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def parse_markdown(self, md_content: str) -> List[CodeBlock]:
        """解析markdown文档，提取代码块和Git信息"""
        code_blocks = []

        # 找到Files部分的开始
        files_start = md_content.find("# Files")
        if files_start == -1:
            print("警告: 未找到'# Files'标题")
            return []

        files_section = md_content[files_start:]

        # 使用正则表达式匹配文件块
        file_pattern = (
            r"## File: (.+?)\n+"
            r"(?:```+(\w*)\n)?(.*?)(?:\n```+)?"
            r"(?:\n+### Git提交历史.*?|"
            r"\n+\*\*Git提交历史:\*\* 无法获取提交历史.*?|"
            r"(?=\n## File:|\Z))"
        )

        matches = re.finditer(file_pattern, files_section, re.DOTALL)

        for match in matches:
            file_path = match.group(1).strip()
            language = match.group(2) or "text"
            content = match.group(3).strip()

            # 查找该文件的Git提交历史
            git_commits = self._extract_git_commits(files_section, file_path)

            # 计算metadata
            token_count = len(self.encoding.encode(content))

            code_block = CodeBlock(
                file_path=file_path,
                content=content,
                language=language,
                git_commits=git_commits,
                metadata={
                    "token_count": token_count,
                    "file_size": len(content),
                    "commit_count": len(git_commits),
                },
            )

            code_blocks.append(code_block)

        return code_blocks

    def _extract_git_commits(self, content: str, file_path: str) -> List[GitCommitInfo]:
        """提取指定文件的Git提交信息"""
        commits = []

        # 找到该文件后的Git提交历史部分
        file_start = content.find(f"## File: {file_path}")
        if file_start == -1:
            return commits

        # 找到下一个文件的开始位置
        next_file_start = content.find("## File:", file_start + 1)
        if next_file_start == -1:
            file_section = content[file_start:]
        else:
            file_section = content[file_start:next_file_start]

        if file_section.find("**Git提交历史:** 无法获取提交历史") != -1:
            return commits

        # 查找Git提交历史
        git_history_start = file_section.find("### Git提交历史")
        if git_history_start == -1:
            return commits

        git_section = file_section[git_history_start:]

        # 匹配提交信息
        commit_pattern = (
            r"#### 提交 \d+\n"
            r"- \*\*提交标识:\*\* ?`?(.+?)`?\n"
            r"- \*\*提交者:\*\* ?(.+?)\n"
            r"- \*\*提交时间:\*\* ?(.+?)\n"
            r"- \*\*提交信息:\*\* ?(.+?)\n"
            r"\n详细改动如下：\n\n```diff\n(.*?)\n```"
        )
        commit_matches = re.finditer(commit_pattern, git_section, re.DOTALL)

        for commit_match in commit_matches:
            commit = GitCommitInfo(
                commit_id=commit_match.group(1).strip(),
                author=commit_match.group(2).strip(),
                date=commit_match.group(3).strip(),
                message=commit_match.group(4).strip(),
                diff_content=commit_match.group(5).strip(),
            )
            commits.append(commit)

        return commits


class DocumentProcessor:
    """文档处理器 - 实现三种处理策略"""

    def __init__(self):
        # 增加chunk_size，减少文档数量
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,  # 增加到2000
            chunk_overlap=200, 
            separators=["\n\n", "\n", " ", ""]
        )

    def create_basic_documents(self, code_blocks: List[CodeBlock]) -> List[Document]:
        """基础版本：仅包含代码内容"""
        documents = []

        for block in code_blocks:
            # 创建基础内容
            content = f"# File: {block.file_path} ({block.language})\n\n"
            content += "## Code Content:\n"
            content += "```" + block.language + "\n"
            content += block.content + "\n"
            content += "```\n\n"

            # 分块处理
            chunks = self.text_splitter.split_text(content)

            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "file_path": block.file_path,
                        "language": block.language,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "strategy": "basic",
                    },
                )
                documents.append(doc)

        return documents

    def create_enhanced_documents(self, code_blocks: List[CodeBlock]) -> List[Document]:
        """增强版本：代码内容 + Git提交历史"""
        documents = []

        for block in code_blocks:
            # 创建增强内容
            content = f"# File: {block.file_path} ({block.language})\n\n"
            content += "## Code Content:\n"
            content += "```" + block.language + "\n"
            content += block.content + "\n"
            content += "```\n\n"

            # 添加Git提交历史
            if block.git_commits:
                content += "## Git Commit History\n\n"
                for i, commit in enumerate(block.git_commits[:5], 1):  # 最多显示5个提交
                    content += f"### Commit {i}\n"
                    content += f"- ID: {commit.commit_id[:8]}\n"
                    content += f"- Author: {commit.author}\n"
                    content += f"- Date: {commit.date}\n"
                    content += f"- Message: {commit.message}\n"
                    content += (
                        f"- Changes:\n```diff\n{commit.diff_content[:500]}...\n```\n\n"
                    )

            # 分块处理
            chunks = self.text_splitter.split_text(content)

            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "file_path": block.file_path,
                        "language": block.language,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "commit_count": len(block.git_commits),
                        "strategy": "enhanced",
                    },
                )
                documents.append(doc)

        return documents

    def create_compressed_documents(
        self, code_blocks: List[CodeBlock]
    ) -> List[Document]:
        """压缩版本：关键代码 + 简化提交信息"""
        documents = []

        for block in code_blocks:
            # 创建压缩内容
            content = f"# File: {block.file_path} ({block.language})\n\n"

            # 压缩代码内容 - 提取关键行
            key_lines = self._extract_key_lines(block.content)
            if key_lines:
                content += "## Key Code Elements:\n"
                content += "\n".join(key_lines[:15]) + "\n\n"  # 最多15行关键代码

            # 压缩Git历史
            if block.git_commits:
                content += "## Recent Changes:\n"
                for commit in block.git_commits[:3]:  # 最多3个提交
                    content += f"- {commit.commit_id[:8]} by {commit.author}: {commit.message[:100]}...\n"
                content += "\n"

            # 分块处理
            chunks = self.text_splitter.split_text(content)

            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "file_path": block.file_path,
                        "language": block.language,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "commit_count": len(block.git_commits),
                        "strategy": "compressed",
                    },
                )
                documents.append(doc)

        return documents

    def _extract_key_lines(self, content: str) -> List[str]:
        """提取代码中的关键行"""
        lines = content.split("\n")
        key_lines = []

        for line in lines:
            stripped = line.strip()
            # 保留关键语句
            if (
                stripped.startswith(("def ", "class ", "function ", "async def "))
                or stripped.startswith(("import ", "from "))
                or stripped.startswith(("@", "export ", "const ", "let ", "var "))
                or stripped.endswith(":")
                or "TODO" in stripped
                or "FIXME" in stripped
                or "NOTE" in stripped
            ):
                key_lines.append(line)

        return key_lines


class LangChainRAGSystem:
    """基于LangChain和ChromaDB的RAG系统"""

    def __init__(self, collection_name: str, model_name: str = "all-MiniLM-L6-v2"):
        self.collection_name = collection_name
        
        # 配置HuggingFace Embeddings以使用本地缓存，避免网络请求
        model_kwargs = {
            'device': 'cpu',  # 使用CPU
            'trust_remote_code': False
        }
        encode_kwargs = {
            'normalize_embeddings': True
        }
        
        # 设置离线模式和本地缓存
        self.embeddings = HuggingFaceEmbeddings(
            model_name=f"sentence-transformers/{model_name}",
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
            cache_folder=os.path.expanduser("~/.cache/huggingface/hub"),
        )

        # 设置ChromaDB持久化目录
        self.persist_directory = f"./chroma_db_{collection_name}"

        # 初始化Chroma向量存储
        self.vectorstore = None

        # if os.path.exists(self.persist_directory):
        #     shutil.rmtree(self.persist_directory)

    def index_documents(self, documents: List[Document]):
        """索引文档到向量数据库"""
        if not documents:
            print("警告: 没有文档需要索引")
            return

        # 创建或加载向量存储
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
        )

        print(f"已索引 {len(documents)} 个文档块到集合 '{self.collection_name}'")

    def _create_vectorstore_from_documents(self, documents: List[Document]):
        """从文档创建新的向量存储"""
        return Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
        )

    def _add_documents_to_vectorstore(self, documents: List[Document]):
        """向现有向量存储添加文档"""
        if self.vectorstore is None:
            raise ValueError("向量存储未初始化，请先调用 _create_vectorstore_from_documents")
        
        self.vectorstore.add_documents(documents)

    def search(self, query: str, k: int = 5) -> List[Dict]:
        """搜索相关文档"""
        if not self.vectorstore:
            try:
                # 尝试加载已存在的向量存储
                self.vectorstore = Chroma(
                    collection_name=self.collection_name,
                    embedding_function=self.embeddings,
                    persist_directory=self.persist_directory,
                )
            except Exception:
                print(f"错误: 向量存储 '{self.collection_name}' 不存在，请先索引文档")
                return []

        # 执行相似性搜索
        docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=k)

        results = []
        for doc, score in docs_with_scores:
            results.append(
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity_score": float(score),
                }
            )

        return results


class RAGComparisonExperiment:
    """RAG对比实验框架"""

    def __init__(self):
        self.parser = MarkdownParser()
        self.processor = DocumentProcessor()

        # 三个不同策略的RAG系统
        self.rag_basic = LangChainRAGSystem("basic_rag")
        self.rag_enhanced = LangChainRAGSystem("enhanced_rag")
        self.rag_compressed = LangChainRAGSystem("compressed_rag")

        self.code_blocks = []

    def setup_experiment(self, markdown_file: str):
        """设置实验环境"""
        print("正在解析markdown文件...")

        with open(markdown_file, "r", encoding="utf-8") as f:
            md_content = f.read()

        # 解析代码块
        self.code_blocks = self.parser.parse_markdown(md_content)
        print(f"解析出 {len(self.code_blocks)} 个代码块")

        if not self.code_blocks:
            print("错误: 没有解析到任何代码块")
            return

        for block in self.code_blocks:
            print(
                f"文件: {block.file_path}, 语言: {block.language}, 提交数: {len(block.git_commits)}, 令牌数: {block.metadata['token_count']}"
            )

        # 创建三种不同的文档集合
        print("创建基础版本文档...")
        basic_docs = self.processor.create_basic_documents(self.code_blocks)
        print(f"基础版本: {len(basic_docs)} 个文档块")

        # for doc in basic_docs:
        #     print(
        #         f"  - {doc.metadata.get('file_path', 'unknown')}\n内容如下:\n{doc.page_content[:100]}...\n"
        #     )

        print("创建增强版本文档...")
        enhanced_docs = self.processor.create_enhanced_documents(self.code_blocks)
        print(f"增强版本: {len(enhanced_docs)} 个文档块")

        print("创建压缩版本文档...")
        compressed_docs = self.processor.create_compressed_documents(self.code_blocks)
        print(f"压缩版本: {len(compressed_docs)} 个文档块")

        # 索引到不同的向量数据库
        print("索引基础版本...")
        self.rag_basic.index_documents(basic_docs)

        print("索引增强版本...")
        self.rag_enhanced.index_documents(enhanced_docs)

        print("索引压缩版本...")
        self.rag_compressed.index_documents(compressed_docs)

        print("实验环境设置完成!")

    def run_comparison(self, test_queries: List[str], k: int = 3) -> Dict:
        """运行对比实验"""
        results = {}

        for query in test_queries:
            print(f"\n测试查询: {query}")
            results[query] = {}

            # 基础版本搜索
            basic_results = self.rag_basic.search(query, k=k)
            results[query]["basic"] = basic_results

            # 增强版本搜索
            enhanced_results = self.rag_enhanced.search(query, k=k)
            results[query]["enhanced"] = enhanced_results

            # 压缩版本搜索
            compressed_results = self.rag_compressed.search(query, k=k)
            results[query]["compressed"] = compressed_results

            # 显示结果概要
            print("基础版本找到的文件:")
            for r in basic_results:
                print(
                    f"  - {r['metadata'].get('file_path', 'unknown')} (相似度: {r['similarity_score']:.3f})"
                )

            print("增强版本找到的文件:")
            for r in enhanced_results:
                print(
                    f"  - {r['metadata'].get('file_path', 'unknown')} (相似度: {r['similarity_score']:.3f})"
                )

            print("压缩版本找到的文件:")
            for r in compressed_results:
                print(
                    f"  - {r['metadata'].get('file_path', 'unknown')} (相似度: {r['similarity_score']:.3f})"
                )

        return results

    def save_results(
        self, results: Dict, output_file: str = "rag_comparison_results.json"
    ):
        """保存实验结果"""
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"实验结果已保存到 {output_file}")


# 使用示例
def main():
    """主函数 - 运行RAG对比实验"""

    # 初始化实验
    experiment = RAGComparisonExperiment()

    # 设置实验环境 (请修改为实际的markdown文件路径)
    markdown_file = "repomix-output-Alma.md"
    experiment.setup_experiment(markdown_file)

    # 定义测试查询
    test_queries = [
        "当前代码库中有哪些依赖是最近才添加的？",
        "Which dependencies in the current code base were added recently?",
        "generate_default_nickname()函数中可能会有什么问题？",
    ]

    # 运行对比实验
    print("开始运行对比实验...")
    results = experiment.run_comparison(test_queries, k=3)

    # 保存结果
    experiment.save_results(results)

    print("\n实验完成!")


if __name__ == "__main__":
    main()
