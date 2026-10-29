"""
End-to-End RAG Application
---------------------------
Loads documents (PDF / TXT / Web URL) -> chunks them -> embeds them ->
stores them in a local Chroma vector DB -> retrieves relevant chunks for
a user query -> generates an answer using Groq (free-tier LLM inference,
serving open models like Llama 3).

Run with:
    streamlit run app.py
"""

import os
import tempfile

import streamlit as st
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

PERSIST_DIR = "./chroma_db"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "llama-3.3-70b-versatile"  

st.set_page_config(page_title="RAG with Groq", page_icon="📚", layout="wide")

@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)


def get_vectorstore():
    """Re-create the Chroma wrapper each call (cheap) but reuse persisted data on disk."""
    return Chroma(
        collection_name="rag_collection",
        embedding_function=get_embeddings(),
        persist_directory=PERSIST_DIR,
    )


def get_llm(api_key: str):
    return ChatOpenAI(
        model=GROQ_MODEL,
        api_key=api_key,
        base_url=GROQ_BASE_URL,
        temperature=0.2,
    )


def load_uploaded_file(uploaded_file):
    """Save an uploaded file to a temp path and load it with the right loader."""
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    if suffix == ".pdf":
        loader = PyPDFLoader(tmp_path)
    elif suffix in (".txt", ".md"):
        loader = TextLoader(tmp_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    docs = loader.load()
    # attach a clean source name instead of the temp path
    for d in docs:
        d.metadata["source"] = uploaded_file.name
    return docs


def load_url(url: str):
    loader = WebBaseLoader(url)
    docs = loader.load()
    for d in docs:
        d.metadata["source"] = url
    return docs


def chunk_documents(docs, chunk_size=800, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def format_docs(docs):
    return "\n\n".join(
        f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
        for d in docs
    )


st.sidebar.title(" Ingest Documents")

api_key = st.sidebar.text_input("Groq API Key", type="password")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF or TXT files", type=["pdf", "txt", "md"], accept_multiple_files=True
)
url_input = st.sidebar.text_input("...or paste a web URL")

chunk_size = st.sidebar.slider("Chunk size", 200, 2000, 800, step=100)
chunk_overlap = st.sidebar.slider("Chunk overlap", 0, 400, 100, step=50)
top_k = st.sidebar.slider("Chunks to retrieve (top-k)", 1, 10, 4)

if st.sidebar.button("Process & Index", type="primary"):
    all_docs = []
    with st.spinner("Loading documents..."):
        for f in uploaded_files or []:
            try:
                all_docs.extend(load_uploaded_file(f))
            except Exception as e:
                st.sidebar.error(f"Failed to load {f.name}: {e}")
        if url_input:
            try:
                all_docs.extend(load_url(url_input))
            except Exception as e:
                st.sidebar.error(f"Failed to load URL: {e}")

    if not all_docs:
        st.sidebar.warning("No documents loaded. Upload a file or enter a URL.")
    else:
        with st.spinner("Chunking & embedding..."):
            chunks = chunk_documents(all_docs, chunk_size, chunk_overlap)
            vectordb = get_vectorstore()
            vectordb.add_documents(chunks)
        st.sidebar.success(f"Indexed {len(chunks)} chunks from {len(all_docs)} document(s).")

if st.sidebar.button(" Clear vector store"):
    vectordb = get_vectorstore()
    vectordb.delete_collection()
    st.sidebar.success("Vector store cleared.")

st.title(" RAG Chatbot — powered by Groq")
st.caption("Upload documents on the left, then ask questions about them below.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask a question about your documents...")

if query:
    if not api_key:
        st.error("Please enter your Groq API key in the sidebar.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    vectordb = get_vectorstore()
    retriever = vectordb.as_retriever(search_kwargs={"k": top_k})

    prompt = ChatPromptTemplate.from_template(
        """You are a helpful assistant answering questions using ONLY the
provided context. If the answer isn't in the context, say you don't know
based on the available documents — do not make something up.

Context:
{context}

Question: {question}

Answer (cite the source filename/URL where relevant):"""
    )

    llm = get_llm(api_key)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant context..."):
            retrieved_docs = retriever.invoke(query)

        with st.expander(f" Retrieved {len(retrieved_docs)} chunk(s)"):
            for i, d in enumerate(retrieved_docs, 1):
                st.markdown(f"**{i}. {d.metadata.get('source', 'unknown')}**")
                st.text(d.page_content[:500] + ("..." if len(d.page_content) > 500 else ""))

        chain = (
            {"context": lambda x: format_docs(retrieved_docs), "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        response_placeholder = st.empty()
        full_response = ""
        with st.spinner("Generating answer..."):
            for token in chain.stream(query):
                full_response += token
                response_placeholder.markdown(full_response + "▌")
        response_placeholder.markdown(full_response)

        import re, io

        def markdown_table_to_html(text):
            """Convert markdown tables in text to HTML tables; leave other text as <p>."""
            lines = text.split("\n")
            html_parts = []
            i = 0
            while i < len(lines):
                line = lines[i]
                # Detect markdown table row
                if line.strip().startswith("|") and line.strip().endswith("|"):
                    table_lines = []
                    while i < len(lines) and lines[i].strip().startswith("|"):
                        table_lines.append(lines[i])
                        i += 1
                    # Build HTML table
                    html_parts.append('<table border="1" style="border-collapse:collapse;width:100%">')
                    for j, tl in enumerate(table_lines):
                        cells = [c.strip() for c in tl.strip().strip("|").split("|")]
                        # skip separator row (--- row)
                        if all(re.match(r"^-+$", c) for c in cells if c):
                            continue
                        tag = "th" if j == 0 else "td"
                        html_parts.append("<tr>" + "".join(f"<{tag} style='padding:4px 8px'>{c}</{tag}>" for c in cells) + "</tr>")
                    html_parts.append("</table>")
                else:
                    if line.strip():
                        html_parts.append(f"<p>{line}</p>")
                    i += 1
            return "\n".join(html_parts)

        html_content = f"""
        <html><head><meta charset='utf-8'>
        <style>
            body {{ font-family: Calibri, Arial, sans-serif; font-size: 11pt; margin: 40px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
            th, td {{ border: 1px solid #888; padding: 6px 10px; text-align: left; }}
            th {{ background-color: #4472C4; color: white; font-weight: bold; }}
            tr:nth-child(even) td {{ background-color: #f2f2f2; }}
            p {{ margin: 6px 0; }}
        </style></head>
        <body>{markdown_table_to_html(full_response)}</body></html>
        """

        st.download_button(
            label=" Download as Word-compatible file (.doc)",
            data=html_content.encode("utf-8"),
            file_name="rag_response.doc",
            mime="application/msword",
        )

        with st.expander(" Copy raw text"):
            st.code(full_response, language=None)

    st.session_state.messages.append({"role": "assistant", "content": full_response})