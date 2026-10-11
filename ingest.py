"""
ingest.py
---------
Clones a GitHub repo (any language) or fetches a website, splits the
content into meaningful chunks, embeds them, and stores them in a
persistent ChromaDB collection.

Usage:
    python ingest.py https://github.com/psf/requests
    python ingest.py https://spring.io/guides/gs/rest-service
"""

import os
import re
import sys
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from git import Repo
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_core.documents import Document

REPOS_DIR = "repos"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "codebase"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


LANGUAGE_EXTENSIONS = {
    ".py": Language.PYTHON,
    ".java": Language.JAVA,
    ".js": Language.JS,
    ".jsx": Language.JS,
    ".ts": Language.TS,
    ".tsx": Language.TS,
    ".go": Language.GO,
    ".rb": Language.RUBY,
    ".php": Language.PHP,
    ".cpp": Language.CPP,
    ".cc": Language.CPP,
    ".cxx": Language.CPP,
    ".hpp": Language.CPP,
    ".c": Language.C,
    ".h": Language.C,
    ".cs": Language.CSHARP,
    ".kt": Language.KOTLIN,
    ".rs": Language.RUST,
    ".scala": Language.SCALA,
    ".swift": Language.SWIFT,
    ".html": Language.HTML,
    ".htm": Language.HTML,
}

# Non-code text files worth indexing too (project config, docs) but that
# don't need a code-aware splitter.
GENERIC_TEXT_EXTENSIONS = {
    ".md", ".txt", ".yaml", ".yml", ".json", ".xml", ".gradle",
    ".properties", ".toml", ".ini", ".cfg", ".sql",
}

SKIP_DIRS = {
    ".git", "venv", "env", "__pycache__", "node_modules", ".venv",
    "target", "build", "dist", ".idea", ".gradle", "bin", "obj",
}


def clone_repo(repo_url: str) -> str:
    """Clone repo_url into REPOS_DIR/<repo_name>, skipping if already cloned."""
    repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
    local_path = os.path.join(REPOS_DIR, repo_name)

    if os.path.exists(local_path):
        print(f"[ingest] {repo_name} already cloned, skipping clone.")
        return local_path

    print(f"[ingest] Cloning {repo_url} ...")
    Repo.clone_from(repo_url, local_path, depth=1)  # shallow clone: fast, no history
    return local_path


def load_code_files(local_path: str) -> list[Document]:
    """
    Walk the repo and load every recognized source/config file as a
    LangChain Document, tagging each with its file extension so it can
    later be chunked with the right language-aware splitter.
    """
    docs = []
    lang_counts = {}

    for root, dirs, files in os.walk(local_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext not in LANGUAGE_EXTENSIONS and ext not in GENERIC_TEXT_EXTENSIONS:
                continue

            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, local_path)
            try:
                with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                if content.strip():
                    docs.append(
                        Document(
                            page_content=content,
                            metadata={"file_path": rel_path, "source": rel_path, "ext": ext},
                        )
                    )
                    lang_counts[ext] = lang_counts.get(ext, 0) + 1
            except Exception as e:
                print(f"[ingest] Skipping {full_path}: {e}")

    print(f"[ingest] Loaded {len(docs)} files.")
    if lang_counts:
        breakdown = ", ".join(f"{ext} x{n}" for ext, n in sorted(lang_counts.items()))
        print(f"[ingest] File types found: {breakdown}")
    else:
        print("[ingest] WARNING: no recognized source files found in this repo.")
    return docs


def chunk_documents(docs: list[Document]) -> list[Document]:
    """
    Split each document with a splitter appropriate to its language,
    grouping by file extension so a Java file is split at method/class
    boundaries, a JS file at function boundaries, etc., rather than
    forcing every language through the Python splitter (or skipping
    non-Python files entirely, which was the original bug).
    """
    by_ext: dict[str, list[Document]] = {}
    for doc in docs:
        by_ext.setdefault(doc.metadata.get("ext", ""), []).append(doc)

    all_chunks: list[Document] = []
    for ext, ext_docs in by_ext.items():
        language = LANGUAGE_EXTENSIONS.get(ext)
        if language is not None:
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=language,
                chunk_size=1000,
                chunk_overlap=100,
            )
        else:
            # Generic text/config files (README, YAML, XML, etc.)
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
        all_chunks.extend(splitter.split_documents(ext_docs))

    # Add a chunk index per file so citations can point to "file.java (part 2)"
    file_chunk_counts: dict[str, int] = {}
    for chunk in all_chunks:
        fp = chunk.metadata["file_path"]
        file_chunk_counts[fp] = file_chunk_counts.get(fp, 0) + 1
        chunk.metadata["chunk_index"] = file_chunk_counts[fp]

    print(f"[ingest] Split into {len(all_chunks)} chunks across {len(by_ext)} file type(s).")
    return all_chunks


def store_in_chroma(chunks: list[Document], repo_name: str, source_type: str = "code"):
    """Embed chunks and persist them in ChromaDB, tagged with the repo/site name."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    for chunk in chunks:
        chunk.metadata["repo"] = repo_name
        chunk.metadata["source_type"] = source_type

    vectordb = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    vectordb.add_documents(chunks)
    print(f"[ingest] Stored {len(chunks)} chunks in Chroma at '{CHROMA_DIR}'.")


def fetch_website(url: str) -> Document:
    """Fetch a webpage and extract its main readable text (strips nav, scripts, ads, etc.)."""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; CodebaseRAGBot/1.0)"}
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "svg", "form"]):
        tag.decompose()

    title = soup.title.string.strip() if soup.title and soup.title.string else url

    main = soup.find("article") or soup.find("main") or soup.body or soup
    text = main.get_text(separator="\n")

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    if not text:
        raise ValueError(f"No readable text found at {url}")

    return Document(
        page_content=text,
        metadata={"file_path": f"{title} ({url})", "source": url},
    )


def chunk_website_documents(docs: list[Document]) -> list[Document]:
    """Split plain webpage text into overlapping chunks (no code-aware splitting needed)."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    file_chunk_counts = {}
    for chunk in chunks:
        fp = chunk.metadata["file_path"]
        file_chunk_counts[fp] = file_chunk_counts.get(fp, 0) + 1
        chunk.metadata["chunk_index"] = file_chunk_counts[fp]

    print(f"[ingest] Split webpage into {len(chunks)} chunks.")
    return chunks


def ingest_website(url: str):
    """Fetch a single webpage, chunk it, and store it in Chroma under its domain name."""
    print(f"[ingest] Fetching {url} ...")
    doc = fetch_website(url)
    chunks = chunk_website_documents([doc])
    domain = urlparse(url).netloc.replace("www.", "")
    store_in_chroma(chunks, repo_name=domain, source_type="website")
    print(f"[ingest] Done. '{url}' is ready to query.")


def is_github_repo_url(url: str) -> bool:
    """True only for a GitHub repo root URL (owner/repo), not a specific file/blob link."""
    parsed = urlparse(url)
    if "github.com" not in parsed.netloc:
        return False
    parts = [p for p in parsed.path.split("/") if p]
    return len(parts) == 2


def ingest_source(url: str):
    """Dispatch to repo ingestion or website ingestion depending on the URL."""
    if is_github_repo_url(url):
        ingest_repo(url)
    else:
        ingest_website(url)


def ingest_repo(repo_url: str):
    local_path = clone_repo(repo_url)
    repo_name = os.path.basename(local_path)
    docs = load_code_files(local_path)
    if not docs:
        print("[ingest] No recognized source files found. Aborting.")
        return
    chunks = chunk_documents(docs)
    store_in_chroma(chunks, repo_name)
    print(f"[ingest] Done. '{repo_name}' is ready to query.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python ingest.py <github_repo_url_or_website_url>")
        sys.exit(1)
    ingest_source(sys.argv[1])