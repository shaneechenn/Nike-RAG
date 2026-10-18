"""
PHASES 2, 4, 5 — the three kinds of search + the AI answer.

- keyword_search() : exact words, via SQLite FTS5 (BM25 ranking).
- semantic_search(): meaning, via sentence-transformers (all-mpnet-base-v2).
- hybrid_search()  : both, merged with Reciprocal Rank Fusion (RRF).
- answer()         : (optional) an LLM writes a friendly recommendation.

Test from the command line:
    python search.py "a good gift for my mom"
"""

import os
import re
import sqlite3

import chromadb
from sentence_transformers import SentenceTransformer

DB_PATH = "nike.db"
CHROMA_DIR = "chroma_db"
MODEL_NAME = "all-mpnet-base-v2"

_model = None
_collection = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = client.get_collection("nike_products")
    return _collection


def _conn():
    return sqlite3.connect(DB_PATH)


_STOPWORDS = {
    "a", "an", "the", "for", "of", "in", "on", "at", "to", "and", "or",
    "is", "are", "was", "were", "be", "been", "am", "i", "you", "he",
    "she", "it", "we", "they", "me", "my", "your", "his", "her", "our",
    "their", "this", "that", "these", "those", "want", "wanna", "need",
    "looking", "please", "some", "best", "good", "great", "show", "find",
    "get", "buy", "give", "can", "with", "what", "whats", "do", "does",
    "should", "would", "could", "like", "just", "any", "something",
    "recommend", "recommendation", "recommendations", "gift", "gifts",
}


def keyword_search(query, k=10):
    """Find products matching the query's content words (OR + BM25).

    Uses OR (not AND) so "shoes for dad" matches products containing
    "shoes", while filler/intent words like "for", "dad" are left to the
    semantic side. BM25 ranks the rare, meaningful words highest.
    """
    words = re.findall(r"[A-Za-z0-9]+", query.lower())
    terms = [w for w in words if w not in _STOPWORDS]
    if not terms:
        return []
    q = " OR ".join(f'"{w}"' for w in terms)
    try:
        conn = _conn()
        rows = conn.execute(
            "SELECT p.id FROM products_fts f JOIN products p ON p.id = f.rowid "
            "WHERE products_fts MATCH ? ORDER BY bm25(products_fts) LIMIT ?",
            (q, k),
        ).fetchall()
        conn.close()
        return [r[0] for r in rows]
    except sqlite3.OperationalError:
        # Malformed query -> no keyword hits (semantic search still works).
        return []


def semantic_search(query, k=10):
    """Return the k products whose *meaning* is closest to the query."""
    qvec = get_model().encode(query).tolist()
    res = get_collection().query(
        query_embeddings=[qvec], n_results=k, include=["metadatas"]
    )
    return [m["pid"] for m in res["metadatas"][0]]


def hybrid_search(query, k=10):
    """Merge keyword + semantic results with Reciprocal Rank Fusion.

    Retrieves a larger semantic pool (top 30) so the fusion has more
    candidates to choose from before trimming down to k.
    """
    kw = keyword_search(query, max(k * 6, 30))
    sem = semantic_search(query, max(k * 6, 30))
    score = {}
    for id_list in (kw, sem):
        for rank, pid in enumerate(id_list):
            score[pid] = score.get(pid, 0) + 1.0 / (60 + rank + 1)
    return sorted(score, key=score.get, reverse=True)[:k]


def get_products(ids):
    """Return full product rows, preserving the order given."""
    conn = _conn()
    rows = [
        conn.execute(
            "SELECT id, title, subtitle, description, brand, color, "
            "price, avg_rating, review_count FROM products WHERE id=?",
            (i,),
        ).fetchone()
        for i in ids
    ]
    conn.close()
    return [r for r in rows if r]


def format_price(p):
    return f"${p:.2f}" if p is not None else None


def answer(question, top_ids, model="gpt-4o-mini"):
    """Ask an LLM to write a recommendation using ONLY the retrieved products.

    Requires the OPENAI_API_KEY environment variable to be set.
    """
    from openai import OpenAI

    products = get_products(top_ids)
    lines = []
    for _, title, _, desc, _, color, price, rating, _ in products:
        bits = [title]
        if color:
            bits.append(color)
        p = format_price(price)
        if p:
            bits.append(p)
        if rating is not None:
            bits.append(f"rated {rating:.1f}")
        lines.append(f"- {', '.join(bits)}: {desc}")
    context = "\n".join(lines)
    prompt = (
        "You are a Nike shopping assistant. Recommend up to 3 products from the "
        "list below for the customer's request, with a one-line reason each. "
        "Only use the products provided.\n\n"
        f"Customer: {question}\n\nProducts:\n{context}"
    )
    client = OpenAI()  # reads OPENAI_API_KEY from the environment
    return client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    ).choices[0].message.content


if __name__ == "__main__":
    import sys

    # Make the console tolerate non-ASCII characters (e.g. ®, é, ★).
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    q = sys.argv[1] if len(sys.argv) > 1 else "good gift for my mom"
    ids = hybrid_search(q, k=5)
    print(f"Query: {q}\n")
    for _, title, subtitle, desc, _, color, price, rating, reviews in get_products(ids):
        meta = []
        if color:
            meta.append(color)
        p = format_price(price)
        if p:
            meta.append(p)
        if rating is not None:
            meta.append(f"rating {rating:.1f}/5 ({reviews or 0} reviews)")
        print(f"- {title}")
        if subtitle:
            print(f"    {subtitle}")
        if meta:
            print("    " + "  |  ".join(meta))
        print(f"    {desc[:140]}")
        print()
