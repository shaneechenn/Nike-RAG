"""
PHASE 6 — the chat interface (Streamlit), styled like ChatGPT.

Run it with:
    streamlit run app.py

If OPENAI_API_KEY is set, an LLM also writes a friendly recommendation;
otherwise you just get the ranked product cards.
"""

import html
import os

import streamlit as st

from search import answer, format_price, get_products, hybrid_search

HERE = os.path.dirname(os.path.abspath(__file__))
USER_AVATAR = os.path.join(HERE, "user_avatar.png")
ASSISTANT_AVATAR = os.path.join(HERE, "assistant_avatar.png")

st.set_page_config(page_title="Nike Assistant", page_icon="👟")

# --- clean, ChatGPT-like look ---
st.markdown(
    """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .block-container {padding-top: 2rem; padding-bottom: 2rem; max-width: 840px;}
        [data-testid="stChatMessage"] {border-radius: 16px;}

        /* Chat input: no red glow on focus, just a subtle gray border */
        [data-testid="stChatInput"]:focus-within {
            border-color: #cbd5e1 !important;
            box-shadow: none !important;
            outline: none !important;
        }
        [data-testid="stChatInputTextArea"]:focus {
            box-shadow: none !important;
            outline: none !important;
        }

        /* Send button: blue instead of red */
        [data-testid="stChatInputSubmitButton"] {
            color: #2563eb !important;
        }
        [data-testid="stChatInputSubmitButton"] svg {
            fill: #2563eb !important;
            stroke: #2563eb !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("## 👟 Nike Assistant")


def render_user(text):
    """Render the user's message in blue."""
    st.markdown(
        f'<span style="color:#2563eb; font-weight:500;">{html.escape(text)}</span>',
        unsafe_allow_html=True,
    )


def render_product(p):
    """Draw one product as a neat card."""
    _, title, subtitle, desc, _, color, price, rating, reviews = p
    with st.container(border=True):
        st.markdown(f"**{title}**")
        meta = []
        if subtitle:
            meta.append(subtitle)
        if color:
            meta.append(color)
        pstr = format_price(price)
        if pstr:
            meta.append(pstr)
        if rating is not None:
            meta.append(f"⭐ {rating:.1f} ({reviews or 0} reviews)")
        if meta:
            st.caption(" · ".join(meta))
        if desc:
            st.markdown(desc[:240] + ("…" if len(desc) > 240 else ""))


def render_assistant(m, expanded):
    """Render an assistant turn: optional written answer + collapsible product list."""
    if m.get("answer"):
        st.markdown(m["answer"])
    if m.get("products"):
        with st.expander(
            f"View {len(m['products'])} recommended products", expanded=expanded
        ):
            for p in m["products"]:
                render_product(p)


if "messages" not in st.session_state:
    st.session_state.messages = []

# --- render chat history (previous answers stay as neat, collapsible cards) ---
for m in st.session_state.messages:
    if m["role"] == "user":
        with st.chat_message("user", avatar=USER_AVATAR):
            render_user(m["content"])
    else:
        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
            render_assistant(m, expanded=False)

# --- handle a new query ---
if q := st.chat_input("What are you looking for?"):
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user", avatar=USER_AVATAR):
        render_user(q)

    ids = hybrid_search(q, k=5)
    products = get_products(ids)

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        answer_text = None
        if os.getenv("OPENAI_API_KEY"):
            try:
                answer_text = answer(q, ids)
                st.markdown(answer_text)
            except Exception as e:  # noqa: BLE001
                st.caption(f"(AI answer unavailable: {e})")
        # Fresh answer starts expanded so results are visible immediately.
        render_assistant({"answer": answer_text, "products": products}, expanded=True)

    st.session_state.messages.append(
        {"role": "assistant", "answer": answer_text, "products": products}
    )
