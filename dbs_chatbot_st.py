"""
DBS RAG Chatbot — Streamlit Edition
=====================================
LangChain RAG chain + Gemini + ChromaDB, with DBS-themed Streamlit UI.

Usage:
    pip install -r requirements.txt
    streamlit run dbs_chatbot.py
"""

import streamlit as st
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# ── Config ─────────────────────────────────────────────────────────────────────
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "dbs_help_support"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "gemini-2.5-flash"
TOP_K = 5

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DBS Help & Support Chatbot",
    page_icon="🏦",
    layout="wide",
)

# ── DBS-themed CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* DBS Red accent */
    .stApp { }
    [data-testid="stSidebar"] {
        background-color: #1a1a1a;
        color: white;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] li,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label {
        color: white !important;
    }
    .main-header {
        background: linear-gradient(135deg, #e31837 0%, #b5132c 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
    }
    .main-header h1 {
        margin: 0;
        font-size: 1.8rem;
    }
    .main-header p {
        margin: 0.3rem 0 0 0;
        opacity: 0.9;
        font-size: 0.95rem;
    }
    .source-box {
        background-color: #f8f9fa;
        border-left: 3px solid #e31837;
        padding: 0.5rem 1rem;
        margin-top: 0.5rem;
        font-size: 0.85rem;
        border-radius: 0 5px 5px 0;
    }
    .stChatMessage [data-testid="stMarkdownContainer"] {
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Load RAG Components (cached so they only load once) ────────────────────────
@st.cache_resource
def load_rag_chain():
    """Load embeddings, vector store, LLM, and build the RAG chain."""
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY", "") or st.secrets.get("GEMINI_API_KEY", "")
    if not api_key:
        st.error("⚠️ GEMINI_API_KEY not found. Create a `.env` file with: GEMINI_API_KEY=your_key_here")
        st.stop()

    # Embeddings + Vector Store
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_PATH,
    )

    # Retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})

    # LLM
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=api_key)

    # Format retrieved docs
    def format_docs(docs):
        formatted = []
        for doc in docs:
            source = doc.metadata.get("url", "")
            title = doc.metadata.get("title", "")
            formatted.append(f"[Source: {title}]\n{doc.page_content}\n(URL: {source})")
        return "\n\n---\n\n".join(formatted)

    # Prompt
    template = """You are a helpful DBS Singapore customer support assistant.
Answer the user's question based ONLY on the following context retrieved from DBS Help & Support articles.

Rules:
- Be concise and practical.
- If the context contains step-by-step instructions, present them clearly.
- Mention relevant channels (digibank mobile app, digibank online, branch, hotline 1800 111 1111).
- Include the source URL(s) at the end of your answer so the user can read more.
- If the context does not contain enough information to answer, say so and suggest the user visit www.dbs.com.sg/personal/support or call the hotline.

Context:
{context}

Question: {query}

Answer:"""

    prompt = ChatPromptTemplate.from_template(template)

    # RAG chain
    rag_chain = (
        {"context": retriever | format_docs, "query": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, llm, vector_store


# Load everything
rag_chain, llm, vector_store = load_rag_chain()
chunk_count = vector_store._collection.count()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("DBS_logo.jpg", width=120)
    st.markdown("---")
    st.markdown("### About This Chatbot")
    st.markdown(
        f"""
        This chatbot answers questions about **DBS Singapore** 
        banking services using **Retrieval-Augmented Generation (RAG)**.

        **How it works:**
        1. Your question is matched against **{chunk_count:,}** 
           knowledge chunks from DBS Help & Support
        2. The top {TOP_K} most relevant chunks are retrieved
        3. **Gemini** generates an answer grounded in those chunks

        **Tech Stack:**
        - ChromaDB (vector database)
        - LangChain (RAG pipeline)
        - Gemini 2.5 Flash (LLM)
        - Streamlit (UI)
        """
    )
    st.markdown("---")
    st.markdown("### Sample Questions")
    sample_questions = [
        "How do I transfer money overseas?",
        "What is PayNow?",
        "How to apply for a credit card?",
        "How to reset my digibank PIN?",
        "What are the home loan interest rates?",
    ]
    for q in sample_questions:
        if st.button(q, key=f"sample_{q}", use_container_width=True):
            st.session_state["sample_question"] = q

    st.markdown("---")
    mode = st.radio("Mode", ["💬 Chat", "🔍 RAG vs LLM"], index=0)


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>Welcome to DBS Help & Support Chatbot</h1>
    <p>Ask anything about DBS Singapore banking services, powered by RAG</p>
</div>
""", unsafe_allow_html=True)


# ── Chat Mode ──────────────────────────────────────────────────────────────────
if mode == "💬 Chat":
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle sample question clicks
    if "sample_question" in st.session_state:
        prompt_text = st.session_state.pop("sample_question")
    else:
        prompt_text = st.chat_input("Ask a question about DBS services...")

    if prompt_text:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": prompt_text})
        with st.chat_message("user"):
            st.markdown(prompt_text)

        # Get and show assistant response
        with st.chat_message("assistant"):
            with st.spinner("Searching DBS knowledge base..."):
                response = rag_chain.invoke(prompt_text)
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

    # Clear chat button
    if st.session_state.messages:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


# ── RAG vs LLM Comparison Mode ────────────────────────────────────────────────
elif mode == "🔍 RAG vs LLM":
    st.markdown("#### Compare answers with and without the DBS knowledge base")
    st.markdown("See why RAG matters： the LLM-only answer relies on general training data, "
                "while RAG grounds its answer in actual DBS documentation.")

    query = st.text_input("Enter your question:", placeholder="e.g. How do I transfer money overseas?")

    if query:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("##### 💡 RAG Answer")
            st.caption("With DBS knowledge base")
            with st.spinner("Retrieving + generating..."):
                rag_answer = rag_chain.invoke(query)
            st.success(rag_answer)

        with col2:
            st.markdown("##### 🗣️ LLM-only Answer")
            st.caption("Gemini without context")
            with st.spinner("Generating..."):
                llm_answer = llm.invoke(query).content

            st.warning(llm_answer)
