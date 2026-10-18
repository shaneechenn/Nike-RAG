"""
Nike product search — keyword-first, semantic re-rank, recipient-aware routing.

- keyword_search()  : FTS5 OR + BM25 (concrete words like "shoes").
- semantic_search() : all-mpnet-base-v2 vector search (meaning).
- _semantic_rank()  : re-rank a candidate set by cosine similarity.
- detect_recipient(): project the query onto the model's gender/age axes
                      (no word lists — generalizes to "aunt", "uncle", ...).
- hybrid_search()   : keyword -> semantic -> cross-encoder -> recipient rank.

Test from the command line:
    python search.py "shoes for dad"
"""

import re
import sqlite3

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

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


# Intent words like "gift" are NOT product attributes — as keywords they would
# only match "Gift Box"/"Gift Card" products instead of things a dad would want.
_STOPWORDS = {
    "a", "an", "the", "for", "of", "in", "on", "at", "to", "and", "or",
    "is", "are", "was", "were", "be", "been", "am", "i", "you", "he",
    "she", "it", "we", "they", "me", "my", "your", "his", "her", "our",
    "their", "this", "that", "these", "those", "want", "wanna", "need",
    "looking", "please", "some", "best", "good", "great", "show", "find",
    "get", "buy", "give", "can", "with", "what", "whats", "do", "does",
    "should", "would", "could", "like", "just", "any", "something",
    "recommend", "recommendation", "recommendations",
    "gift", "gifts", "presents", "present", "birthday",
}


def keyword_search(query, k=10):
    """Match products by the query's content words (OR + BM25)."""
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
        return []


def semantic_search(query, k=10):
    """Return the k products whose *meaning* is closest to the query."""
    qvec = get_model().encode(query).tolist()
    res = get_collection().query(
        query_embeddings=[qvec], n_results=k, include=["metadatas"]
    )
    return [m["pid"] for m in res["metadatas"][0]]


def _semantic_rank(query, pids, k=10):
    """Re-rank candidate ids by cosine similarity to the query embedding."""
    if not pids:
        return []
    qvec = np.array(get_model().encode(query))
    res = get_collection().get(ids=[str(p) for p in pids], include=["embeddings"])
    id_to_vec = {
        res["ids"][i]: np.array(res["embeddings"][i]) for i in range(len(res["ids"]))
    }
    scored = []
    for pid in pids:
        vec = id_to_vec.get(str(pid))
        if vec is None:
            continue
        sim = float(np.dot(qvec, vec) / (np.linalg.norm(qvec) * np.linalg.norm(vec) + 1e-9))
        scored.append((pid, sim))
    scored.sort(key=lambda x: -x[1])
    return [p for p, _ in scored][:k]


# --- recipient-aware routing via the model's own geometry (no word lists) ---

_GENDER_PAIRS = [
    ("man", "woman"), ("he", "she"), ("father", "mother"),
    ("husband", "wife"), ("brother", "sister"), ("boy", "girl"),
    ("male", "female"),
]
_AGE_PAIRS = [
    ("adult", "child"), ("man", "boy"), ("woman", "girl"),
    ("parent", "baby"), ("grownup", "kid"),
]

_axes = {}


def _get_axes():
    if "gender" not in _axes:
        model = get_model()

        def axis(pairs):
            vecs = [model.encode(a) - model.encode(b) for a, b in pairs]
            ax = np.mean(vecs, axis=0)
            return ax / (np.linalg.norm(ax) + 1e-9)

        _axes["gender"] = axis(_GENDER_PAIRS)
        _axes["age"] = axis(_AGE_PAIRS)
    return _axes["gender"], _axes["age"]


def detect_recipient(query):
    """Infer the recipient by projecting each query WORD onto the model's
    gender/age axes. Checks words individually so neutral words ("volleyball",
    "red", "shoes") don't dilute a strong recipient word ("dad", "son").
    Works for any word the model knows — no word lists.
    """
    g_axis, a_axis = _get_axes()
    model = get_model()
    words = re.findall(r"[A-Za-z0-9]+", query.lower())
    best_g = 0.0
    best_a = 0.0
    for w in words:
        v = np.array(model.encode(w))
        g = float(np.dot(v, g_axis))
        a = float(np.dot(v, a_axis))
        if a < best_a:
            best_a = a
        if abs(g) > abs(best_g):
            best_g = g
    if best_a < -0.15:    # a word strongly means a child
        return "KIDS"
    if best_g > 0.06:     # a word strongly means male
        return "MEN"
    if best_g < -0.18:    # a word strongly means female
        return "WOMEN"
    return None


_RECIPIENT_GROUPS = {
    "MEN": {"MEN"},
    "WOMEN": {"WOMEN"},
    "KIDS": {"BOYS", "GIRLS"},
}


def _gender_ids(recipient):
    """Product ids whose Nike gender label matches the recipient."""
    target = _RECIPIENT_GROUPS[recipient]
    conn = _conn()
    rows = conn.execute("SELECT id, gender FROM products").fetchall()
    conn.close()
    ids = []
    for pid, gs in rows:
        if gs:
            g = set(gs.split("|"))
            if g & target:
                ids.append(pid)
    return ids


def _recipient_rank(pids, recipient):
    """Re-rank using Nike's own gender labels: match > unisex > mismatch."""
    target = _RECIPIENT_GROUPS[recipient]
    conn = _conn()
    gender_map = dict(conn.execute("SELECT id, gender FROM products").fetchall())
    conn.close()

    def score(pid):
        gs = gender_map.get(pid)
        if not gs:
            return 0.0  # unisex
        g = set(gs.split("|"))
        if g & target:
            return 1.0 if g <= target else 0.5  # exact > mixed
        return -1.0  # mismatch

    indexed = [(pid, i) for i, pid in enumerate(pids)]
    indexed.sort(key=lambda x: (-score(x[0]), x[1]))
    return [pid for pid, _ in indexed]


# --- cross-encoder (general relevance re-ranking) ---

_cross_model = None


def get_cross_model():
    global _cross_model
    if _cross_model is None:
        _cross_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_model


def _rerank(query, pids):
    """Re-score candidates with a cross-encoder for general relevance."""
    if not pids:
        return pids
    model = get_cross_model()
    conn = _conn()
    ph = ",".join("?" for _ in pids)
    rows = conn.execute(
        f"SELECT id, title, subtitle FROM products WHERE id IN ({ph})", pids
    ).fetchall()
    conn.close()
    text_map = {r[0]: f"{r[1]} {r[2]}".strip() for r in rows}
    pairs = [(query, text_map.get(pid, "")) for pid in pids]
    scores = model.predict(pairs)
    scored = sorted(zip(pids, scores), key=lambda x: -x[1])
    return [pid for pid, _ in scored]


def _single_search(query, k):
    """Search a single (non-OR) query: keyword -> semantic -> cross-encoder -> recipient."""
    pool = max(k * 6, 40)
    recipient = detect_recipient(query)

    # 1) keyword retrieval
    kw = keyword_search(query, pool)

    # 2) semantic re-ranking of the candidates
    if kw:
        ranked = _semantic_rank(query, kw, k=pool)
    else:
        ranked = semantic_search(query, pool)

    # 3) if a recipient is implied, add gender-matched candidates
    if recipient:
        extra = _semantic_rank(query, _gender_ids(recipient), k=30)
        seen = set(ranked)
        for pid in extra:
            if pid not in seen:
                seen.add(pid)
                ranked.append(pid)

    # 4) cross-encoder re-rank for general relevance
    ranked = _rerank(query, ranked[:30])

    # 5) recipient-aware ranking: matching gender first
    if recipient:
        ranked = _recipient_rank(ranked, recipient)

    return ranked[:k]


def _split_intents(query):
    """Split 'basketball shoes or volleyball shoes' into separate intents.

    Handles a shared trailing noun: 'red or blue shoes' -> 'red shoes' + 'blue shoes'.
    """
    parts = re.split(r"\s+(?:or|and|vs|versus)\s+", query, flags=re.IGNORECASE)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) <= 1:
        return [query]
    tail = parts[-1].split()[-1] if len(parts[-1].split()) >= 2 else ""
    out = []
    for p in parts:
        if tail and not p.endswith(tail):
            out.append((p + " " + tail).strip())
        else:
            out.append(p)
    return out


def _round_robin(lists, k):
    """Interleave results so every intent contributes at least one product."""
    result = []
    seen = set()
    i = 0
    while len(result) < k and any(i < len(l) for l in lists):
        for l in lists:
            if i < len(l) and l[i] not in seen:
                seen.add(l[i])
                result.append(l[i])
                if len(result) >= k:
                    break
        i += 1
    return result


def hybrid_search(query, k=10):
    """Keyword-first search, with multi-intent support for 'X or Y' queries."""
    intents = _split_intents(query)
    if len(intents) == 1:
        return _single_search(intents[0], k)
    per = [_single_search(intent, k) for intent in intents]
    return _round_robin(per, k)


def get_products(ids):
    """Return full product rows, preserving the order given."""
    conn = _conn()
    rows = [
        conn.execute(
            "SELECT id, title, subtitle, description, brand, color, "
            "price, url FROM products WHERE id=?",
            (i,),
        ).fetchone()
        for i in ids
    ]
    conn.close()
    return [r for r in rows if r]


def format_price(p):
    return f"${p:.2f}" if p is not None else None


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    q = sys.argv[1] if len(sys.argv) > 1 else "shoes for dad"
    ids = hybrid_search(q, k=5)
    print(f"Query: {q}\n")
    for _, title, subtitle, desc, _, color, price, _url in get_products(ids):
        meta = []
        if color:
            meta.append(color)
        p = format_price(price)
        if p:
            meta.append(p)
        print(f"- {title}")
        if subtitle:
            print(f"    {subtitle}")
        if meta:
            print("    " + "  |  ".join(meta))
        print(f"    {desc[:140]}")
        print()
