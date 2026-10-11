
from urllib.parse import urlparse

import streamlit as st

from ingest import ingest_source, is_github_repo_url
from query import get_vectordb, build_chain

st.set_page_config(page_title="RAG Assistant", page_icon="🧠", layout="wide")
st.title("🧠 RAG Assistant")
st.caption("Chat with any public GitHub repo or website — powered by LangChain + ChromaDB")


if "ingested_sources" not in st.session_state:
    st.session_state.ingested_sources = []  # list of (label, repo_key)
if "messages" not in st.session_state:
    st.session_state.messages = []


with st.sidebar:
    st.header("1. Add a source")
    source_url = st.text_input(
        "GitHub repo URL or website URL",
        placeholder="https://github.com/psf/requests  or  https://example.com/article",
    )

    if st.button("Ingest", type="primary") and source_url:
        is_repo = is_github_repo_url(source_url)
        kind = "repo" if is_repo else "website"
        with st.spinner(f"Fetching and indexing this {kind} ... this can take a minute"):
            try:
                ingest_source(source_url)
                if is_repo:
                    key = source_url.rstrip("/").split("/")[-1].replace(".git", "")
                else:
                    key = urlparse(source_url).netloc.replace("www.", "")
                label = f"{key} ({kind})"
                if label not in [l for l, _ in st.session_state.ingested_sources]:
                    st.session_state.ingested_sources.append((label, key))
                st.success(f"'{key}' indexed successfully.")
            except Exception as e:
                st.error(f"Failed to ingest: {e}")

    st.header("2. Filter (optional)")
    labels = ["All indexed sources"] + [l for l, _ in st.session_state.ingested_sources]
    source_choice_label = st.selectbox("Ask about a specific source", options=labels)
    # map the chosen label back to its underlying key for filtering
    key_lookup = dict(st.session_state.ingested_sources)
    source_choice = key_lookup.get(source_choice_label)

# --- Main chat area ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask something, or say 'summarize this' ...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and thinking..."):
            try:
                vectordb = get_vectordb()
                chain, retriever = build_chain(vectordb, source_choice)

                answer = chain.invoke(question)
                sources = retriever.invoke(question)

                st.markdown(answer)

                if sources:
                    with st.expander("📄 Sources"):
                        for doc in sources:
                            fp = doc.metadata.get("file_path", "unknown")
                            idx = doc.metadata.get("chunk_index", "?")
                            st.markdown(f"- `{fp}` (chunk {idx})")

                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                error_msg = f"Something went wrong: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})