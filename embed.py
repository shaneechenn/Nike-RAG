"""
PHASE 3 — Turn each product's text into an "embedding" (a vector) and store
it in Chroma, our vector database.

Uses the stronger all-mpnet-base-v2 model (768-dim) for better semantic quality.

Run once:
    python embed.py
"""

import sqlite3

import chromadb
from sentence_transformers import SentenceTransformer

DB_PATH = "nike.db"
CHROMA_DIR = "chroma_db"
MODEL_NAME = "all-mpnet-base-v2"
BATCH = 32


def main():
    print(f"Loading embedding model {MODEL_NAME} (first run downloads ~420 MB)...")
    model = SentenceTransformer(MODEL_NAME)

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

    embs = []
    for i in range(0, len(docs), BATCH):
        embs.extend(model.encode(docs[i : i + BATCH]).tolist())

    metas = [{"pid": pid} for pid, _ in rows]
    col.upsert(ids=ids, documents=docs, embeddings=embs, metadatas=metas)
    print(f"Done. {len(ids)} vectors stored in {CHROMA_DIR}.")


if __name__ == "__main__":
    main()
