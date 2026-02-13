"""
DBS RAG Ingestion Pipeline (LangChain version)
================================================
Loads scraped DBS Help & Support data, chunks it intelligently,
and stores embeddings in ChromaDB via LangChain.

Run this AFTER dbs_scraper.py has produced dbs_knowledge_base.json.

Usage:
    pip install langchain langchain-chroma langchain-huggingface sentence-transformers
    python dbs_ingest.py

Output:
    - ./chroma_db/  (persistent ChromaDB database)
"""

import json
import re
import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
KNOWLEDGE_BASE_PATH = "dbs_knowledge_base.json"
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "dbs_help_support"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Chunking parameters
MAX_CHUNK_SIZE = 1000       # chars per chunk (target)
CHUNK_OVERLAP = 150         # overlap between consecutive chunks
MIN_CHUNK_SIZE = 50         # skip chunks smaller than this


# ── Chunking Logic (smarter splitting) ───────────────────────────────

def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def chunk_by_sentences(text: str, max_size: int = MAX_CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into chunks at sentence boundaries with overlap."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(current_chunk) + len(sentence) > max_size and current_chunk:
            chunks.append(current_chunk.strip())
            overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
            current_chunk = overlap_text + " " + sentence
        else:
            current_chunk = (current_chunk + " " + sentence).strip()

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return [c for c in chunks if len(c) >= MIN_CHUNK_SIZE]


def create_langchain_docs_from_article(article: dict) -> list[Document]:
    """
    Create LangChain Document objects from a single scraped article.

    Using a chunking strategy instead of RecursiveCharacterTextSplitter:
    1. Each FAQ pair → one Document (question + answer kept together)
    2. Step-by-step instructions → grouped Documents
    3. Heading sections → Document(s), split by sentence boundary if too large
    4. Full text fallback → sentence-based chunking
    5. Important notes → separate Documents
    """
    docs = []
    url = article["url"]
    title = article.get("title", "")
    category = article.get("category", "")
    context_prefix = f"[{category}] {title}\n"

    # ── 1. FAQ Pairs ──
    for faq in article.get("faq_pairs", []):
        q = faq.get("question", "").strip()
        a = faq.get("answer", "").strip()
        if q and a:
            docs.append(Document(
                page_content=f"{context_prefix}Q: {q}\nA: {a}",
                metadata={
                    "url": url, "title": title,
                    "category": category, "chunk_type": "faq",
                    "question": q, "source": url,
                },
            ))

    # ── 2. Step-by-step Instructions ──
    steps = article.get("steps", [])
    if steps:
        step_group = context_prefix + "How to (step-by-step):\n"
        for step in steps:
            if len(step_group) + len(step) > MAX_CHUNK_SIZE and len(step_group) > len(context_prefix) + 30:
                docs.append(Document(
                    page_content=step_group.strip(),
                    metadata={
                        "url": url, "title": title,
                        "category": category, "chunk_type": "steps",
                        "source": url,
                    },
                ))
                step_group = context_prefix + "How to (continued):\n"
            step_group += step + "\n"

        if len(step_group.strip()) > len(context_prefix) + 30:
            docs.append(Document(
                page_content=step_group.strip(),
                metadata={
                    "url": url, "title": title,
                    "category": category, "chunk_type": "steps",
                    "source": url,
                },
            ))

    # ── 3. Sections (by heading) ──
    for section in article.get("sections", []):
        heading = section.get("heading", "")
        content = section.get("content", "")
        if not content:
            continue

        section_text = f"{context_prefix}{heading}\n{content}"

        if len(section_text) <= MAX_CHUNK_SIZE:
            docs.append(Document(
                page_content=section_text,
                metadata={
                    "url": url, "title": title,
                    "category": category, "chunk_type": "section",
                    "section_heading": heading, "source": url,
                },
            ))
        else:
            for sc in chunk_by_sentences(content):
                docs.append(Document(
                    page_content=f"{context_prefix}{heading}\n{sc}",
                    metadata={
                        "url": url, "title": title,
                        "category": category, "chunk_type": "section",
                        "section_heading": heading, "source": url,
                    },
                ))

    # ── 4. Full text fallback (if no structured content found) ──
    if not docs:
        full_text = article.get("full_text", "")
        if full_text:
            for tc in chunk_by_sentences(full_text):
                docs.append(Document(
                    page_content=f"{context_prefix}{tc}",
                    metadata={
                        "url": url, "title": title,
                        "category": category, "chunk_type": "text",
                        "source": url,
                    },
                ))

    # ── 5. Important notes ──
    for note in article.get("notes", []):
        if len(note) >= MIN_CHUNK_SIZE:
            docs.append(Document(
                page_content=f"{context_prefix}Important: {note}",
                metadata={
                    "url": url, "title": title,
                    "category": category, "chunk_type": "note",
                    "source": url,
                },
            ))

    return docs


# ── Main Ingestion ─────────────────────────────────────────────────────────────

def ingest():
    """Load scraped data, chunk it, and store in ChromaDB via LangChain."""

    # 1. Load scraped data
    kb_path = Path(KNOWLEDGE_BASE_PATH)
    if not kb_path.exists():
        logger.error(f"Knowledge base not found at {KNOWLEDGE_BASE_PATH}")
        logger.error("Run dbs_scraper.py first!")
        return

    with open(kb_path, "r", encoding="utf-8") as f:
        articles = json.load(f)

    logger.info(f"Loaded {len(articles)} articles from {KNOWLEDGE_BASE_PATH}")

    # 2. Chunk all articles into LangChain Documents
    all_docs = []
    for article in articles:
        all_docs.extend(create_langchain_docs_from_article(article))

    logger.info(f"Created {len(all_docs)} chunks from {len(articles)} articles")

    # Chunk type distribution
    type_counts = {}
    for d in all_docs:
        ct = d.metadata.get("chunk_type", "unknown")
        type_counts[ct] = type_counts.get(ct, 0) + 1
    logger.info(f"Chunk types: {type_counts}")

    # 3. Set up embeddings (model: all-MiniLM-L6-v2)
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # 4. Store in ChromaDB via LangChain 
    logger.info(f"Storing in ChromaDB at {CHROMA_DB_PATH}...")
    vector_store = Chroma.from_documents(
        documents=all_docs,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH,
    )

    logger.info(f"✅ Ingestion complete! {len(all_docs)} chunks stored in ChromaDB.")

    # 5. Sanity test
    print("\n" + "=" * 60)
    print("SANITY TEST")
    print("=" * 60)

    test_queries = [
        "How to transfer money overseas?",
        "What is PayNow?",
        "How to reset my digibank PIN?",
    ]

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    for query in test_queries:
        results = retriever.invoke(query)
        print(f"\n🔍 Query: {query}")
        for i, doc in enumerate(results):
            print(f"   [{i+1}] ({doc.metadata.get('chunk_type', '')}) {doc.metadata.get('title', '')[:60]}")
            print(f"       {doc.page_content[:120]}...")

    print("\n" + "=" * 60)
    print(f"ChromaDB ready at: {CHROMA_DB_PATH}")
    print(f"Collection: {COLLECTION_NAME}")
    print("=" * 60)


if __name__ == "__main__":
    ingest()
