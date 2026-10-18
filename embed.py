"""
PHASE 3 — Turn each product's text into an "embedding" (a vector) and store
it in Chroma, our vector database.

Uses Chroma's built-in ONNX embedding model (all-MiniLM-L6-v2) — no torch /
sentence-transformers needed, so it loads much faster.

Run once:
    python embed.py
"""

import sqlite3

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

DB_PATH = "nike.db"
CHROMA_DIR = "chroma_db"
BATCH = 32


def main():
    print("Loading embedding model (ONNX all-MiniLM-L6-v2; first run downloads ~80 MB)...")
    ef = DefaultEmbeddingFunction()

    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT id, combined_text FROM products").fetchall()
    conn.close()
    print(f"Found {len(rows)} products to embed.")

    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Start fresh each run: upsert only overwrites matching ids, it never removes
    # leftover ids from a previous dataset, which would leave stale vectors behind.
    try:
        client.delete_collection("nike_products")
    except Exception:
        pass
    col = client.create_collection("nike_products")

    ids = [str(pid) for pid, _ in rows]
    docs = [text for _, text in rows]

    # Embed in small batches (the ONNX model has a max batch size).
    embs = []
    for i in range(0, len(docs), BATCH):
        embs.extend(ef(docs[i : i + BATCH]))

    metas = [{"pid": pid} for pid, _ in rows]
    col.upsert(ids=ids, documents=docs, embeddings=embs, metadatas=metas)
    print(f"Done. {len(ids)} vectors stored in {CHROMA_DIR}.")


if __name__ == "__main__":
    main()
