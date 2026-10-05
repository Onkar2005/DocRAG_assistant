# PDF Question-Answering System using RAG

Upload a PDF, ask questions in plain English, and get answers grounded in the
document's own content (with the source pages shown).

## How it works

```
PDF -> extract & clean text -> split into chunks -> embeddings -> FAISS index
                                                                     |
Question -> embedding -> find most similar chunks -> LLM answers using only those chunks
```

| Step | File | What it does |
|------|------|--------------|
| 1 | `rag/pdf_loader.py` | Extracts text with pypdf and cleans it |
| 2 | `rag/chunker.py` | Splits text into ~200-word chunks with 40-word overlap |
| 3 | `rag/embedder.py` | Converts chunks to vectors (sentence-transformers) |
| 4 | `rag/vector_store.py` | Stores vectors in FAISS and does similarity search |
| 5 | `rag/llm.py` | Sends the top chunks + question to the LLM |
| - | `rag/pipeline.py` | Connects all steps |
| - | `app.py` | Streamlit interface |

## Run it

```bash
python -m venv venv
venv\Scripts\activate          # Windows   (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env         # Mac/Linux: cp .env.example .env
# open .env and add ONE API key (or leave empty to use the free local model)
streamlit run app.py
```

## Choosing the language model

- `ANTHROPIC_API_KEY` set -> uses Claude
- `OPENAI_API_KEY` set -> uses OpenAI
- No key -> uses the free local `flan-t5-base` model (downloads once, answers are simpler)

## Limits

- Scanned PDFs (images) need OCR first; this version reads text-based PDFs only.
- The index lives in memory and is rebuilt when you upload a new file.

## Ideas to extend

Multiple PDFs, saving the index to disk, chat history, a re-ranking step, answer evaluation.
