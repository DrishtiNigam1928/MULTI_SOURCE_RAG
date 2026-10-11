"""
query.py
--------
Loads the persisted Chroma collection and answers natural-language
questions about the ingested codebase/website, citing the source for
every answer.

"""

import os
import sys

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "codebase"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

PROMPT_TEMPLATE = """You are a knowledgeable assistant answering questions using the context below,
which may come from source code, documentation, or webpages. Answer using ONLY the
context below. If the context doesn't contain enough information, say so plainly
instead of guessing. If asked to summarize, give a clear, well-organized summary
of the relevant context rather than a short one-liner.

Context:
{context}

Question: {question}

Answer clearly and concisely. Refer to specific files, sections, or pages where relevant.
"""


def get_vectordb():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )


def format_docs_with_sources(docs):
    """Join retrieved chunks into a single context string, labeled by file."""
    blocks = []
    for doc in docs:
        fp = doc.metadata.get("file_path", "unknown")
        idx = doc.metadata.get("chunk_index", "?")
        blocks.append(f"# Source: {fp} (chunk {idx})\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)


def build_chain(vectordb, repo_filter: str | None = None, k: int = 5):
    search_kwargs = {"k": k}
    if repo_filter:
        search_kwargs["filter"] = {"repo": repo_filter}

    retriever = vectordb.as_retriever(search_kwargs=search_kwargs)

    # NOTE: llama-3.1-8b-instant was decommissioned by Groq on 2026-08-16.
    # openai/gpt-oss-20b is Groq's official recommended replacement
    # (same "fast/instant" tier). If you want higher-quality answers at the
    # cost of some speed, swap this for "openai/gpt-oss-120b" instead.
    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    chain = (
        {"context": retriever | format_docs_with_sources, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever


def ask(question: str, repo_filter: str | None = None):
    vectordb = get_vectordb()
    chain, retriever = build_chain(vectordb, repo_filter)

    answer = chain.invoke(question)
    sources = retriever.invoke(question)

    print("\n=== ANSWER ===")
    print(answer)
    print("\n=== SOURCES ===")
    for doc in sources:
        print(f"- {doc.metadata.get('file_path')} (chunk {doc.metadata.get('chunk_index')})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python query.py "your question here"')
        sys.exit(1)
    ask(" ".join(sys.argv[1:]))