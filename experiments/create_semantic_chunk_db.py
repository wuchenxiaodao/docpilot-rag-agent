"""DocPilot — 语义章节 Chunk 实验建库脚本。

基于章节标题切分，而非 RecursiveCharacterTextSplitter。
仅操作 ./chroma_db_qwen3_semantic_chunks，不触碰 Baseline。
"""
import argparse
import os
import shutil
import sys

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

# 允许以文件方式直接运行（python experiments/create_semantic_chunk_db.py），
# 而不仅是模块方式（python -m experiments.create_semantic_chunk_db）。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.semantic_chunker import split_by_sections

load_dotenv()

DB_NAME = "./chroma_db_qwen3_semantic_chunks"


def create_semantic_chunk_db(folder_path: str, rebuild: bool = False):
    model_path = os.path.join(
        os.path.expanduser("~"),
        "Models",
        "Qwen3-Embedding-0.6B",
    )

    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Embedding model not found: {model_path}")

    # 安全处理已有数据库
    if os.path.exists(DB_NAME):
        if rebuild:
            shutil.rmtree(DB_NAME)
            print(f"Rebuilding: deleted existing database at {DB_NAME}")
        else:
            print(
                f"Database already exists at {DB_NAME}. "
                "Use --rebuild to delete and recreate."
            )
            sys.exit(0)

    embeddings = HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )

    chroma = Chroma(
        embedding_function=embeddings,
        persist_directory=DB_NAME,
    )

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if not filename.endswith(".pdf"):
            continue

        loader = PyPDFLoader(file_path)
        documents = loader.load()

        # 将 Document 列表转为 pages 元组
        pages = [
            (doc.page_content, doc.metadata.get("page", 0) + 1, doc.metadata.get("source", ""))
            for doc in documents
        ]

        chunks = split_by_sections(pages)

        # 转回 langchain Document 写入 Chroma
        for chunk in chunks:
            doc = Document(
                page_content=chunk["page_content"],
                metadata=chunk["metadata"],
            )
            chunk_id = chroma.add_documents([doc])
            if chunk_id:
                print(f"Chunk added with ID: {chunk_id}")
            else:
                print("Failed to add chunk")

        print(f"Document {filename} added to database ({len(chunks)} semantic chunks).")

    print(f"Semantic chunk database created and saved in {DB_NAME}.")
    return chroma


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create semantic-chunk Chroma database for DocPilot experiment."
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete existing semantic chunk database before rebuilding",
    )
    args = parser.parse_args()

    chroma = create_semantic_chunk_db(
        folder_path="./data",
        rebuild=args.rebuild,
    )
    retriever = chroma.as_retriever(search_kwargs={"k": 3})

    query = "What's my company's mission and values"
    similar_docs = retriever.invoke(query)

    for i, doc in enumerate(similar_docs, start=1):
        print(f"\nResult {i}:\n{doc.page_content}\nTags: {doc.metadata}")