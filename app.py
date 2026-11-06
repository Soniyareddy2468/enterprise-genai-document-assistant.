import os
import streamlit as st
from dotenv import load_dotenv

from document_engine import extract_pdf_pages, chunk_pages, retrieve_chunks, answer_with_openai

load_dotenv()
st.set_page_config(page_title="Enterprise GenAI Document Assistant", page_icon="📚", layout="wide")
st.title("📚 Enterprise GenAI Document Assistant")
st.caption("Ask questions about your PDF documents. Answers include page references.")

with st.sidebar:
    st.header("How it works")
    st.markdown("1. Upload PDFs.\n2. Extract and chunk text.\n3. Retrieve relevant passages.\n4. Optionally generate a grounded answer.")
    use_llm = st.toggle("Use OpenAI to generate answers", value=bool(os.getenv("OPENAI_API_KEY")))
    if use_llm and not os.getenv("OPENAI_API_KEY"):
        st.warning("Add OPENAI_API_KEY to GitHub Codespaces Secrets to enable generated answers.")
    st.caption("Without an API key, evidence-only retrieval still works.")

uploaded_files = st.file_uploader("Upload PDF documents", type=["pdf"], accept_multiple_files=True)
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "doc_names" not in st.session_state:
    st.session_state.doc_names = []

if uploaded_files and st.button("Process documents", type="primary"):
    total_bytes = sum(len(f.getvalue()) for f in uploaded_files)
    if total_bytes > 20 * 1024 * 1024:
        st.error("Please keep the combined upload size under 20 MB.")
    else:
        pages = []
        names = []
        with st.spinner("Reading and indexing documents..."):
            for uploaded in uploaded_files:
                pages.extend(extract_pdf_pages(uploaded.getvalue(), uploaded.name))
                names.append(uploaded.name)
            st.session_state.chunks = chunk_pages(pages)
            st.session_state.doc_names = names
        if st.session_state.chunks:
            st.success(f"Processed {len(names)} file(s) into {len(st.session_state.chunks)} searchable chunks.")
        else:
            st.warning("No selectable text was found. Scanned PDFs need OCR before upload.")

if st.session_state.doc_names:
    st.info("Loaded: " + ", ".join(st.session_state.doc_names))
    question = st.text_input("Ask a question", placeholder="What are the main project risks?")
    top_k = st.slider("Number of passages to retrieve", 1, 6, 4)
    if st.button("Ask", disabled=not question.strip()):
        matches = retrieve_chunks(question, st.session_state.chunks, top_k=top_k)
        if not matches:
            st.warning("No relevant text found. Try a more specific question.")
        elif use_llm and os.getenv("OPENAI_API_KEY"):
            with st.spinner("Generating a source-grounded answer..."):
                answer = answer_with_openai(question, matches)
            st.subheader("Answer")
            st.write(answer)
        else:
            st.subheader("Relevant evidence")
            st.write("LLM generation is off. Review these passages as evidence.")
            for item in matches:
                st.markdown(f"**{item['source']} — page {item['page']}**")
                st.write(item["text"])
                st.divider()
        with st.expander("View retrieved source passages"):
            for item in matches:
                st.markdown(f"**{item['source']} — page {item['page']} (relevance: {item['score']:.3f})**")
                st.write(item["text"])
else:
    st.markdown("### Start here\nUpload a PDF and select **Process documents**. Try public annual reports, product manuals, or non-confidential project documents.")

st.caption("Prototype: local TF-IDF retrieval, not a production vector database. Validate answers before business use.")
