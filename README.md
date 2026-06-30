# RAG App (LangChain + Streamlit + Groq)

A minimal but complete end-to-end Retrieval-Augmented Generation app.

## Pipeline

1. **Load** — PDFs (`PyPDFLoader`), text/markdown (`TextLoader`), or a web page (`WebBaseLoader`)
2. **Chunk** — `RecursiveCharacterTextSplitter`, splitting on paragraph/sentence boundaries first
3. **Embed** — local, free `sentence-transformers/all-MiniLM-L6-v2` model (no API key needed for this step)
4. **Store** — `Chroma`, persisted to `./chroma_db` so your index survives restarts
5. **Retrieve** — top-k similarity search against the query embedding
6. **Generate** — retrieved chunks are stuffed into a prompt and sent to **Groq**, via LangChain's OpenAI-compatible client pointed at `https://api.groq.com/openai/v1`

## Setup

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

You'll need a free Groq API key from https://console.groq.com/keys — paste it into the sidebar field when the app launches (it's not stored anywhere, just used in-session).

## How to use it

1. Upload one or more PDF/TXT files, and/or paste a URL, in the sidebar
2. Click **Process & Index** — this runs steps 1-4 above
3. Ask a question in the chat box at the bottom
4. Expand "Retrieved chunks" under each answer to see exactly what context the model was given — this is the best way to debug bad answers (usually it's a retrieval problem, not a generation problem)

## Things worth experimenting with once it's running

- **Chunk size/overlap** (sidebar sliders) — smaller chunks improve retrieval precision but lose surrounding context; larger chunks do the opposite. Watch how answers change.
- **top-k** — more retrieved chunks isn't always better; irrelevant chunks can confuse the model. Try k=2 vs k=8 on the same question.
- **Swap the embedding model** — try `BAAI/bge-small-en-v1.5` or `intfloat/e5-small-v2` in `EMBED_MODEL_NAME` and compare retrieval quality on the same docs.
- **Add re-ranking** — after the initial vector search, re-score the top ~20 chunks with a cross-encoder (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) and keep the best k. This is usually the single biggest quality improvement after basic RAG is working.
- **Hybrid search** — combine this vector search with a keyword search (BM25) for queries with exact terms/names that embeddings sometimes miss.

## Notes on the Groq integration

Groq's API is OpenAI-compatible, so `ChatOpenAI` from `langchain-openai` works directly — just pointed at a different `base_url` and `model` name. Groq currently has a free tier with generous rate limits and serves fast open models like Llama 3.3. Check https://console.groq.com/docs/models for the current list of available model names if `llama-3.3-70b-versatile` in `app.py` doesn't match what's available.