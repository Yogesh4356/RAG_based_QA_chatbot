# 📄 RAG Document Q&A with Query Expansion, RRF Hybrid Reranking & Session Memory

An end-to-end **Retrieval-Augmented Generation (RAG)** system built with:
- **LangChain**
- **Ollama LLMs + Embeddings**
- **Chroma Vector DB (persistent storage)**
- **Cross-Encoder + BM25 hybrid reranker (RRF)**
- **Multi-query expansion** for better recall
- **Session memory (chat history)** to keep context across turns
- **Streamlit UI** for interactive Q&A

---

## ✨ Features
✅ Upload any **PDF document**  
✅ Automatic **text extraction, cleaning, and chunking**  
✅ **Persistent ChromaDB collections** (no repeated embedding on reload)  
✅ **Multi-query expansion** → improves recall  
✅ **Hybrid Reranking (RRF)** → combines **Cross-Encoder (dense)** + **BM25 (sparse)** for better results  
✅ **RunnableWithMessageHistory memory** → remembers your conversation in-session  
✅ Simple & clean **Streamlit UI** with expansions, top chunks, and final grounded answer  

---

## 🏗️ Architecture Flow

```

```
     ┌─────────────┐
     │   PDF File  │
     └──────┬──────┘
            │ Extract
            ▼
    ┌──────────────┐
    │ Preprocessing│
    └──────┬───────┘
           │ Chunk + Embed
           ▼
  ┌───────────────────┐
  │ Persistent Chroma │
  └──────┬────────────┘
         │
   ┌─────▼───────┐
   │ Query Input │
   └─────┬───────┘
         │ Expand queries (LLM)
         ▼
    ┌───────────────┐
    │ Multi-Retriever│
    └──────┬────────┘
           │ Collect candidates
           ▼
   ┌───────────────────┐
   │ RRF Hybrid Rerank │
   └──────┬────────────┘
          │ Top chunks
          ▼
 ┌──────────────────────┐
 │ Ollama LLM (Context) │
 └─────────┬────────────┘
           │
           ▼
    ┌──────────────┐
    │ Final Answer │
    └──────────────┘
```

````

---

## 🎛️ Usage

1. **Upload a PDF** → text will be extracted, cleaned, chunked, and embedded.

   * Collection is persisted in `./db` (named after the file).
   * On re-upload, system **reuses embeddings** (no repeated computation).

2. **Ask a Question** →

   * System will generate paraphrases of your query.
   * Retrieve chunks with Chroma (dense), expand recall with BM25, then rerank via **RRF (Cross-Encoder + BM25)**.
   * Answer strictly from context.

3. **Conversation Memory** →

   * System remembers your previous questions & answers in the current session.
   * You’ll see the **chat history** at the bottom.
