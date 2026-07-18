import math
import os
import re

import numexpr
from langchain_chroma import Chroma
from langchain_core.tools import BaseTool, tool
from langchain_huggingface import HuggingFaceEmbeddings


def calculator_func(expression: str) -> str:
    """Calculates a math expression using numexpr.

    Useful for when you need to answer questions about math using numexpr.
    This tool is only for math questions and nothing else. Only input
    math expressions.

    Args:
        expression (str): A valid numexpr formatted math expression.

    Returns:
        str: The result of the math expression.
    """

    try:
        local_dict = {"pi": math.pi, "e": math.e}
        output = str(
            numexpr.evaluate(
                expression.strip(),
                global_dict={},  # restrict access to globals
                local_dict=local_dict,  # add common mathematical functions
            )
        )
        return re.sub(r"^\[|\]$", "", output)
    except Exception as e:
        raise ValueError(
            f'calculator("{expression}") raised error: {e}.'
            " Please try again with a valid numerical expression"
        )


calculator: BaseTool = tool(calculator_func)
calculator.name = "Calculator"


# Format retrieved documents
def format_contexts(docs):
    parts = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "Unknown")
        parts.append(f"Document {i}\nSource: {source}\nContent: {doc.page_content}")
    return "\n\n".join(parts)


def _get_embedding_model_path() -> str:
    path = os.environ.get(
        "EMBEDDING_MODEL_PATH",
        os.path.join(os.path.expanduser("~"), "Models", "Qwen3-Embedding-0.6B"),
    )
    if not os.path.isdir(path):
        raise RuntimeError(
            f"Embedding model directory not found: {path}. "
            "Set EMBEDDING_MODEL_PATH env var or place the model at the default location."
        )
    return path


def _get_chroma_db_path() -> str:
    return os.environ.get("CHROMA_DB_PATH", "./chroma_db_qwen3_test")


def load_chroma_db():
    model_path = _get_embedding_model_path()
    db_path = _get_chroma_db_path()

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True},
            query_encode_kwargs={
                "normalize_embeddings": True,
                "prompt_name": "query",
            },
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to initialize HuggingFaceEmbeddings with model at {model_path}."
        ) from e

    chroma_db = Chroma(persist_directory=db_path, embedding_function=embeddings)
    retriever = chroma_db.as_retriever(search_kwargs={"k": 3})
    return retriever


def database_search_func(query: str) -> str:
    """Searches the configured DocPilot PDF/DOCX knowledge base via ChromaDB.

    Returns relevant text fragments and source metadata from indexed documents.
    """
    retriever = load_chroma_db()

    documents = retriever.invoke(query)

    context_str = format_contexts(documents)

    return context_str


database_search: BaseTool = tool(database_search_func)
database_search.name = "Database_Search"
