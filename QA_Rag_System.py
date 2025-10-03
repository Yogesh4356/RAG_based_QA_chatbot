import streamlit as st
import re, os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
import numpy as np

# ---------------- PDF Loader ----------------
def load_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# ---------------- Text Preprocessing ----------------
def preprocess_text(text: str) -> str:
    # remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    # remove non-ascii
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    # normalize punctuation spacing
    text = re.sub(r'\s([?.!,:;])', r'\1', text)
    # lowercase
    text = text.strip().lower()
    return text

# ---------------- Build / Load Retriever ----------------
def build_or_load_retriever(text, collection_name, persist_dir="db"):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # check if collection exists
    if os.path.exists(os.path.join(persist_dir, collection_name)):
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_dir
        )
        print(f"Loaded existing collection: {collection_name}")
    else:
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_text(text)

        vectorstore = Chroma.from_texts(
            chunks,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=persist_dir
        )
        vectorstore.persist()
        print(f"Created new collection: {collection_name}")

    return vectorstore.as_retriever(search_kwargs={"k": 5})

# ---------------- Query Expansion ----------------
llm = OllamaLLM(model="llama2")
expansion_prompt = ChatPromptTemplate.from_template(
    "Generate 3 alternative queries that mean the same as: {query}"
)

def expand_query(query):
    expansion = llm.invoke(expansion_prompt.format_messages(query=query))
    expanded = [q.strip("-• ") for q in expansion.split("\n") if q.strip()]
    return [query] + expanded

# ---------------- Multi-query Retrieval ----------------
def multi_query_retrieve(query, retriever):
    queries = expand_query(query)
    all_docs = []
    for q in queries:
        docs = retriever.get_relevant_documents(q)
        all_docs.extend(docs)
    # deduplicate docs removal
    unique_docs = {doc.page_content: doc for doc in all_docs}
    return list(unique_docs.values()), queries

# ---------------- RRF Reranker (Cross-Encoder + BM25) ----------------
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rrf_rerank(query, docs, top_k=5, k=60):
    if not docs:
        return []

    # --- Cross-Encoder Ranking ---
    pairs = [(query, d.page_content) for d in docs]
    dense_scores = cross_encoder.predict(pairs)
    dense_ranking = np.argsort(-dense_scores)  # descending
    dense_ranks = {docs[idx]: rank for rank, idx in enumerate(dense_ranking)}

    # --- BM25 Ranking ---
    tokenized_corpus = [d.page_content.split() for d in docs]
    bm25 = BM25Okapi(tokenized_corpus)
    sparse_scores = bm25.get_scores(query.split())
    sparse_ranking = np.argsort(-sparse_scores)
    sparse_ranks = {docs[idx]: rank for rank, idx in enumerate(sparse_ranking)}

    # --- RRF Fusion ---
    fused_scores = {}
    for d in docs:
        rank_dense = dense_ranks[d]
        rank_sparse = sparse_ranks[d]
        score = (1 / (k + rank_dense)) + (1 / (k + rank_sparse))
        fused_scores[d] = score

    reranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in reranked[:top_k]]

# ---------------- Memory with Runnable ----------------
if "store" not in st.session_state:
    st.session_state.store = {}

def get_session_history(session_id: str):
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

qa_prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. 
Use the following context to answer the question.

Context:
{context}

Question: {query}

Answer only from the context above. If not found, say "I don't know".
""")

qa_chain = qa_prompt | llm | StrOutputParser()

qa_with_history = RunnableWithMessageHistory(
    qa_chain,
    get_session_history,
    input_messages_key="query",
    history_messages_key="history",
    output_messages_key="answer",
)

# ---------------- Final QA ----------------
def answer_query(query, retriever, session_id="default", top_k=5):
    candidates, expansions = multi_query_retrieve(query, retriever)
    top_docs = rrf_rerank(query, candidates, top_k=top_k)

    context = "\n\n".join(d.page_content for d in top_docs)

    response = qa_with_history.invoke(
        {"query": query, "context": context},
        config={"configurable": {"session_id": session_id}}
    )
    return response, expansions, top_docs

# ---------------- Streamlit UI ----------------
st.title("📄 RAG with Query Expansion + RRF Hybrid Reranking + Runnable Memory")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
if uploaded_file:
    text = load_pdf(uploaded_file)
    text = preprocess_text(text)

    # use file name as collection name
    collection_name = os.path.splitext(uploaded_file.name)[0]
    retriever = build_or_load_retriever(text, collection_name=collection_name)

    query = st.text_input("Ask a question:")
    top_k = st.slider("Number of chunks to consider (Top K)", 1, 10, 5)

    if query:
        with st.spinner("Retrieving and answering..."):
            answer, expansions, top_docs = answer_query(query, retriever, session_id="user1", top_k=top_k)

        st.subheader("🔍 Query Expansions")
        for q in expansions:
            st.write("- " + q)

        st.subheader("📑 Top Retrieved Chunks (RRF Reranked)")
        for i, doc in enumerate(top_docs, 1):
            with st.expander(f"Chunk {i}"):
                st.write(doc.page_content)

        st.subheader("🤖 Final Answer")
        st.write(answer)

        st.subheader("🧠 Conversation History")
        history = get_session_history("user1")
        for msg in history.messages:
            st.write(f"**{msg.type.upper()}:** {msg.content}")
