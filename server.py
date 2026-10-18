"""
Backend API for the Nike assistant (FastAPI + the Python search engine).

Run:
    uvicorn server:app --host 127.0.0.1 --port 8501
"""

import os
import re

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

import search

app = FastAPI()

_FILLERS = {
    "nike", "the", "a", "an", "i", "im", "want", "wanna", "need", "what",
    "whats", "is", "are", "for", "me", "my", "some", "please", "show",
    "find", "looking", "to", "of", "in", "and", "or", "with", "good",
    "best", "great", "get", "buy", "give", "can", "you", "your",
    "recommend", "recommendation", "recommendations", "any", "something",
}


def make_template_answer(query: str, n: int) -> str:
    """A friendly intro line derived from the query (no LLM needed)."""
    words = re.findall(r"[a-z0-9]+", query.lower())
    topic = " ".join(w for w in words if w not in _FILLERS)
    suffix = f" for {topic}" if topic else ""
    if n == 0:
        return 'Hmm, I couldn\'t find a match for that. Try "running shoes" or "a gift for mom".'
    if n == 1:
        return f"Okay, here's 1 recommendation{suffix}!"
    return f"Okay, here are {n} recommendations{suffix}!"


class Query(BaseModel):
    query: str


@app.get("/")
def index():
    """Serve the HTML/CSS/JS frontend."""
    return FileResponse("index.html")


@app.get("/user_avatar.png")
def user_avatar():
    return FileResponse("user_avatar.png")


@app.get("/assistant_avatar.png")
def assistant_avatar():
    return FileResponse("assistant_avatar.png")


@app.post("/search")
def do_search(q: Query):
    """Run the hybrid search and return products + a friendly text reply."""
    ids = search.hybrid_search(q.query, k=5)
    products = search.get_products(ids)
    results = []
    for pid, title, subtitle, desc, brand, color, price, rating, reviews in products:
        results.append(
            {
                "id": pid,
                "title": title,
                "subtitle": subtitle,
                "description": desc,
                "brand": brand,
                "color": color,
                "price": price,
                "rating": rating,
                "reviews": reviews,
            }
        )

    # Friendly reply: use OpenAI if a key is set, otherwise a template line.
    answer_text = make_template_answer(q.query, len(results))
    if os.getenv("OPENAI_API_KEY"):
        try:
            answer_text = search.answer(q.query, ids)
        except Exception:
            pass

    return {"products": results, "answer": answer_text}
