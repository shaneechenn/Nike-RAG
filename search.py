"""
PHASES 2, 4, 5 — the three kinds of search + the AI answer.

- keyword_search() : exact words, via SQLite FTS5 (BM25 ranking).
- semantic_search(): meaning, via Chroma (ONNX all-MiniLM-L6-v2).
- hybrid_search()  : both, merged with Reciprocal Rank Fusion (RRF).
- answer()         : (optional) an LLM writes a friendly recommendation.

Test from the command line:
    python search.py "a good gift for my mom"
"""

import os
import re
import sqlite3

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

DB_PATH = "nike.db"
CHROMA_DIR = "chroma_db"

_ef = None
_collection = None


def get_embedding_function():
    global _ef
    if _ef is None:
        _ef = DefaultEmbeddingFunction()
    return _ef


def get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = client.get_collection("nike_products")
    return _collection


def _conn():
    return sqlite3.connect(DB_PATH)


def keyword_search(query, k=10):
    """Find products containing the literal words of the query."""
    words = re.findall(r"[A-Za-z0-9]+", query)
    if not words:
        return []
    terms = " AND ".join(f'"{w}"' for w in words)
    try:
        conn = _conn()
        rows = conn.execute(
            "SELECT p.id FROM products_fts f JOIN products p ON p.id = f.rowid "
            "WHERE products_fts MATCH ? ORDER BY bm25(products_fts) LIMIT ?",
            (terms, k),
        ).fetchall()
        conn.close()
        return [r[0] for r in rows]
    except sqlite3.OperationalError:
        # Malformed query -> no keyword hits (semantic search still works).
        return []


def semantic_search(query, k=10):
    """Find products whose *meaning* is closest to the query."""
    ef = get_embedding_function()
    qvec = ef([query])[0]
    res = get_collection().query(
        query_embeddings=[qvec], n_results=k, include=["metadatas"]
    )
    return [m["pid"] for m in res["metadatas"][0]]


_MALE = ["dad", "daddy", "father", "husband", "boyfriend", "him", "his", "men",
         "mens", "man", "son", "brother", "boy", "male", "guy"]
_FEMALE = ["mom", "mommy", "mother", "mum", "wife", "girlfriend", "her", "hers",
           "women", "womens", "woman", "daughter", "sister", "girl", "female"]
_KIDS = ["kid", "kids", "child", "toddler", "baby", "infant", "youth", "newborn"]


def detect_query_gender(query):
    """Infer a gender intent from the query (e.g. 'dad' -> men, 'mom' -> women)."""
    t = " " + (query or "").lower() + " "
    if any(w in t for w in _FEMALE):
        return "women"
    if any(w in t for w in _MALE):
        return "men"
    if any(w in t for w in _KIDS):
        return "kids"
    return None


def _gender_rerank(pids, q_gender):
    """Push matching-gender (and unisex) products to the top, mismatches down."""
    conn = _conn()
    gender_map = dict(conn.execute("SELECT id, gender FROM products").fetchall())
    conn.close()

    def key(pid):
        g = gender_map.get(pid)
        return 0 if (g == q_gender or g is None) else 1

    return sorted(pids, key=key)


def hybrid_search(query, k=10):
    """Merge keyword + semantic results with RRF, then gender-aware re-rank."""
    kw = keyword_search(query, k)
    sem = semantic_search(query, k)
    score = {}
    for id_list in (kw, sem):
        for rank, pid in enumerate(id_list):
            score[pid] = score.get(pid, 0) + 1.0 / (60 + rank + 1)
    ranked = sorted(score, key=score.get, reverse=True)

    q_gender = detect_query_gender(query)
    if q_gender:
        ranked = _gender_rerank(ranked, q_gender)

    return ranked[:k]


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
