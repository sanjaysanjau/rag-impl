import streamlit as st

from config import settings
from llm import generate_answer
from pdf_processor import chunk_text, extract_text
from vector_store import query_similar, upsert_chunks

st.set_page_config(
    page_title="PDF Knowledge Base Chat",
    page_icon="📄",
    layout="wide",
)

# ── Session state defaults ────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# ── Sidebar — PDF Upload ──────────────────────────────────────────────────────
with st.sidebar:
    st.title("📄 PDF Upload")
    st.markdown("Upload one or more PDFs to build your knowledge base.")

    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file is not None:
        if uploaded_file.name not in st.session_state.uploaded_files:
            with st.spinner(f"Processing **{uploaded_file.name}** ..."):
                try:
                    file_bytes = uploaded_file.read()
                    text = extract_text(file_bytes)

                    if not text.strip():
                        st.error("This PDF has no extractable text. It may be a scanned image.")
                    else:
                        chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
                        upsert_chunks(chunks, source=uploaded_file.name)
                        st.session_state.uploaded_files.append(uploaded_file.name)
                        st.success(f"Indexed **{len(chunks)}** chunks from `{uploaded_file.name}`")
                except Exception as e:
                    st.error(f"Upload failed: {e}")
        else:
            st.info(f"`{uploaded_file.name}` is already indexed.")

    if st.session_state.uploaded_files:
        st.markdown("---")
        st.markdown("**Indexed Documents**")
        for name in st.session_state.uploaded_files:
            st.markdown(f"- ✅ {name}")

    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Main — Chat Interface ─────────────────────────────────────────────────────
st.title("💬 PDF Knowledge Base Chat")

if not st.session_state.uploaded_files:
    st.info("Upload a PDF from the sidebar to get started.")
else:
    st.markdown(
        f"Chatting against **{len(st.session_state.uploaded_files)}** "
        f"document(s): {', '.join(f'`{f}`' for f in st.session_state.uploaded_files)}"
    )

st.markdown("---")

# Render existing chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources", expanded=False):
                for src in msg["sources"]:
                    st.markdown(f"- {src}")

# Chat input
question = st.chat_input(
    "Ask a question about your documents...",
    disabled=not st.session_state.uploaded_files,
)

if question:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Generate and show assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            try:
                results = query_similar(question)

                if not results:
                    answer = "I couldn't find relevant content. Please upload a PDF first."
                    sources = []
                else:
                    context_chunks = [r["text"] for r in results]
                    sources = [f"{r['source']} (score: {r['score']:.3f})" for r in results]
                    answer = generate_answer(question, context_chunks)

                st.markdown(answer)

                if sources:
                    with st.expander("Sources", expanded=False):
                        for src in sources:
                            st.markdown(f"- {src}")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })

            except Exception as e:
                error_msg = f"Error generating answer: {e}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "sources": [],
                })
