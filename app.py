"""Simple Streamlit UI: upload a PDF, ask questions, see answers with sources."""
import streamlit as st
from dotenv import load_dotenv

from rag.embedder import Embedder
from rag.pipeline import RAGPipeline

load_dotenv()
st.set_page_config(page_title="PDF Q&A (RAG)", page_icon="📄")
st.title("📄 Ask questions about your PDF")


@st.cache_resource
def get_embedder() -> Embedder:
    return Embedder()   # loaded once, reused across reruns


uploaded = st.file_uploader("Upload a PDF", type="pdf")

if uploaded:
    # Rebuild the index only when a new file is uploaded
    if st.session_state.get("file_id") != (uploaded.name, uploaded.size):
        with st.spinner("Reading and indexing your PDF..."):
            pipeline = RAGPipeline(get_embedder())
            try:
                n = pipeline.ingest(uploaded)
            except ValueError as e:
                st.error(str(e))
                st.stop()
        st.session_state.pipeline = pipeline
        st.session_state.file_id = (uploaded.name, uploaded.size)
        st.success(f"Indexed {n} chunks. Ask your question below.")

    question = st.text_input("Your question")
    if question:
        with st.spinner("Searching and generating answer..."):
            result = st.session_state.pipeline.ask(question)
        st.subheader("Answer")
        st.write(result["answer"])
        with st.expander("Sources used (retrieved passages)"):
            for s in result["sources"]:
                st.markdown(f"**Page {s['page']}** (similarity {s['score']:.2f})")
                st.caption(s["text"])
else:
    st.info("Upload a PDF to get started.")
