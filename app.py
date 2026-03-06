"""
app.py — Streamlit UI for Multimodal RAG
==========================================
Run: streamlit run app.py
"""

import gc
import tempfile
from pathlib import Path
from typing import Dict, Any

import streamlit as st

import config
from rag_pipeline import MultimodalRAGPipeline

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=config.PAGE_TITLE,
    page_icon=config.PAGE_ICON,
    layout=config.LAYOUT,
)

# ── Session state init ────────────────────────────────────────────────────────
def init_session():
    if "pipeline"      not in st.session_state:
        st.session_state.pipeline      = None
    if "chat_history"  not in st.session_state:
        st.session_state.chat_history  = []
    if "initialized"   not in st.session_state:
        st.session_state.initialized   = False

init_session()

# ── Load pipeline (cached so it only loads once) ──────────────────────────────
@st.cache_resource(show_spinner="🔧 Initializing RAG pipeline...")
def load_pipeline() -> MultimodalRAGPipeline:
    return MultimodalRAGPipeline()

# ── Helpers ───────────────────────────────────────────────────────────────────
def save_upload(uploaded_file) -> Path:
    """Save a Streamlit uploaded file to a temp path and return it."""
    tmp_dir = Path(tempfile.gettempdir()) / "rag_uploads"
    tmp_dir.mkdir(exist_ok=True)
    dest = tmp_dir / uploaded_file.name
    dest.write_bytes(uploaded_file.getbuffer())
    return dest

def show_processing_results(results: Dict[str, Any]):
    """Display processing stats in a 4-column layout."""
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📄 Documents", results["processed_documents"])
    c2.metric("🖼️ Images",    results["processed_images"])
    c3.metric("🎵 Audio",     results["processed_audio"])
    c4.metric("🗂️ Chunks",    results["stored_chunks"])

    if results["errors"]:
        with st.expander("⚠️ Errors"):
            for err in results["errors"]:
                st.warning(err)

# ── Main App ──────────────────────────────────────────────────────────────────
def main():
    # Header
    st.title("🔍 Multimodal RAG")
    st.caption(
        f"Powered by **{config.DEFAULT_LLM_MODEL}** via Ollama · "
        f"Embeddings: `{config.EMBEDDING_MODEL.split('/')[-1]}`"
    )

    # Load pipeline
    pipeline = load_pipeline()
    st.session_state.pipeline = pipeline

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ System Status")
        try:
            status = pipeline.get_system_status()

            doc_count = status["vector_store"]["document_count"]
            st.metric("📚 Chunks in DB", doc_count)
            st.metric("🧠 LLM Model",    status["llm"]["current_model"])
            st.metric("⚡ Embed Device", status.get("device", "cpu").upper())

            available = status["llm"]["available_models"]
            if available:
                with st.expander("Available Ollama models"):
                    for m in available:
                        st.code(m, language=None)

            st.divider()

            if st.button("🗑️ Clear Knowledge Base", use_container_width=True):
                if pipeline.clear_database():
                    st.success("Knowledge base cleared!")
                    st.session_state.chat_history = []
                    gc.collect()
                    st.rerun()

        except Exception as e:
            st.error(f"Status error: {e}")

        st.divider()
        st.caption(
            "💡 **Model upgrade tip:**\n\n"
            "To use a better model:\n"
            "1. `ollama pull <model>`\n"
            "2. Edit `DEFAULT_LLM_MODEL` in `config.py`\n"
            "3. Restart the app"
        )

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab_upload, tab_chat, tab_history = st.tabs([
        "📁 Upload & Process",
        "💬 Ask Questions",
        "📜 Chat History",
    ])

    # ── Upload Tab ────────────────────────────────────────────────────────────
    with tab_upload:
        st.subheader("Upload Documents")
        st.info(
            f"Supported: PDF, DOCX, TXT, JPG/PNG, WAV/MP3 · "
            f"Max {config.MAX_FILE_SIZE_MB} MB per file · "
            "Upload files one by one for best memory performance."
        )

        uploaded_files = st.file_uploader(
            "Choose files",
            type=["pdf", "docx", "txt", "jpg", "jpeg", "png", "wav", "mp3"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            # File size check
            oversized = [
                f.name for f in uploaded_files
                if f.size / 1024 / 1024 > config.MAX_FILE_SIZE_MB
            ]
            if oversized:
                for name in oversized:
                    st.error(f"❌ '{name}' exceeds {config.MAX_FILE_SIZE_MB} MB limit")
                uploaded_files = [f for f in uploaded_files if f.name not in oversized]

            if uploaded_files and st.button("🚀 Process Files", type="primary", use_container_width=True):
                file_infos = []
                for uf in uploaded_files:
                    saved = save_upload(uf)
                    if saved:
                        file_infos.append({"path": str(saved), "name": uf.name})

                if file_infos:
                    with st.spinner("Processing files..."):
                        results = pipeline.process_and_store_files(file_infos)

                    st.success("✅ Processing complete!")
                    show_processing_results(results)
                    gc.collect()

    # ── Chat Tab ──────────────────────────────────────────────────────────────
    with tab_chat:
        st.subheader("Ask Questions")

        # Show existing chat history
        for entry in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(entry["question"])
            with st.chat_message("assistant"):
                st.write(entry["answer"])
                if entry.get("citations"):
                    with st.expander(f"📎 Sources ({entry['num_sources']})"):
                        for c in entry["citations"]:
                            st.markdown(
                                f"**[{c['index']}]** `{c['source']}` · "
                                f"similarity: {c['similarity']:.3f}"
                            )

        # Input
        question = st.chat_input("Ask something about your documents...")

        if question:
            with st.chat_message("user"):
                st.write(question)

            with st.chat_message("assistant"):
                with st.spinner(f"Thinking with {config.DEFAULT_LLM_MODEL}..."):
                    result = pipeline.query(question)

                answer = result.get("answer", "No answer generated.")
                st.write(answer)

                if result.get("citations"):
                    with st.expander(f"📎 Sources ({result['num_sources']})"):
                        for c in result["citations"]:
                            st.markdown(
                                f"**[{c['index']}]** `{c['source']}` · "
                                f"type: `{c['type']}` · "
                                f"similarity: {c['similarity']:.3f}"
                            )

                if result["status"] not in ("success", "no_docs"):
                    st.warning(f"Status: {result['status']}")

            # Save to history
            st.session_state.chat_history.append({
                "question":   question,
                "answer":     answer,
                "citations":  result.get("citations", []),
                "num_sources": result.get("num_sources", 0),
            })

    # ── History Tab ───────────────────────────────────────────────────────────
    with tab_history:
        st.subheader("Chat History")
        if not st.session_state.chat_history:
            st.info("No questions asked yet.")
        else:
            if st.button("🗑️ Clear History"):
                st.session_state.chat_history = []
                st.rerun()

            for i, entry in enumerate(reversed(st.session_state.chat_history), 1):
                with st.expander(f"Q{i}: {entry['question'][:80]}..."):
                    st.markdown(f"**Question:** {entry['question']}")
                    st.markdown(f"**Answer:** {entry['answer']}")
                    if entry.get("citations"):
                        st.markdown("**Sources:**")
                        for c in entry["citations"]:
                            st.caption(f"  [{c['index']}] {c['source']} (sim: {c['similarity']:.3f})")


if __name__ == "__main__":
    main()
