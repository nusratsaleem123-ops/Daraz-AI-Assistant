"""
app.py
------
Daraz Customer Support Operation Assistant

A Streamlit chat app that:
  - Loads a PRE-BUILT FAISS index + metadata.json (never re-embeds or
    re-processes any PDFs at runtime).
  - Lets the agent restrict search to one knowledge-base section
    (Returns, Delivery, Refunds, Sellers, Payments, Customer Support).
  - Embeds only the user's query (same embedding model used at ingest
    time) to search the index.
  - Sends the retrieved chunks to a Groq-hosted LLM ("openai/gpt-oss-120b")
    to produce a grounded answer.
  - Reads the Groq API key from Streamlit secrets only — it is never
    typed into or displayed in any input box.

Expected folder layout (same directory as this file, or set FAISS_INDEX_DIR):
    faiss_index/
        index.faiss
        metadata.json

Secrets required (Streamlit Cloud: Settings > Secrets, or locally in
.streamlit/secrets.toml):
    GROQ_API_KEY = "your-groq-api-key"

Run:
    streamlit run app.py
"""

import os
import json

import numpy as np
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FAISS_INDEX_DIR = os.environ.get("FAISS_INDEX_DIR", "faiss_index")
INDEX_PATH = os.path.join(FAISS_INDEX_DIR, "index.faiss")
METADATA_PATH = os.path.join(FAISS_INDEX_DIR, "metadata.json")

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"   # must match the model used in ingest.py
GROQ_MODEL_NAME = "openai/gpt-oss-120b"

TOP_K = 5                 # chunks to feed the LLM
OVER_FETCH_K = 50         # candidates pulled before department filtering

SECTIONS = [
    "All Sections",
    "Returns",
    "Delivery",
    "Refunds",
    "Sellers",
    "Payments",
    "Customer Support",
]

DARAZ_ORANGE = "#F85606"
DARAZ_DARK = "#1A1A1A"


# ---------------------------------------------------------------------------
# Page setup + branding
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Daraz Support Assistant",
    page_icon="🛍️",
    layout="wide",
)

st.markdown(
    f"""
    <style>
        .stApp {{
            background-color: #FAFAFA;
        }}
        [data-testid="stSidebar"] {{
            background-color: {DARAZ_DARK};
        }}
        [data-testid="stSidebar"] * {{
            color: #FFFFFF !important;
        }}
        .daraz-header {{
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 14px 20px;
            background: linear-gradient(90deg, {DARAZ_ORANGE}, #FF8A3D);
            border-radius: 10px;
            margin-bottom: 18px;
        }}
        .daraz-header h1 {{
            color: white;
            font-size: 22px;
            margin: 0;
        }}
        .daraz-header p {{
            color: #FFF3EA;
            margin: 0;
            font-size: 13px;
        }}
        .source-tag {{
            display: inline-block;
            background-color: #FFE9DC;
            color: {DARAZ_ORANGE};
            border-radius: 6px;
            padding: 2px 8px;
            font-size: 12px;
            font-weight: 600;
            margin-right: 6px;
        }}
        .stChatMessage {{
            border-radius: 12px;
        }}
        div[data-baseweb="radio"] label {{
            color: #FFFFFF !important;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="daraz-header">
        <div style="font-size:32px;">🛍️</div>
        <div>
            <h1>Daraz Support Assistant</h1>
            <p>Internal knowledge-base assistant for Customer Support agents</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached resource loaders — index, metadata, embedding model, Groq client
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading knowledge base index...")
def load_index_and_metadata():
    if not os.path.exists(INDEX_PATH) or not os.path.exists(METADATA_PATH):
        return None, None

    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # id -> metadata dict, for O(1) lookup after a FAISS search
    metadata_by_id = {item["id"]: item for item in metadata}
    return index, metadata_by_id


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_resource(show_spinner=False)
def load_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)


index, metadata_by_id = load_index_and_metadata()
embedder = load_embedding_model()
groq_client = load_groq_client()


# ---------------------------------------------------------------------------
# Sidebar — knowledge base sections + status
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📚 Knowledge Base")
    st.caption("Restrict search to one section, or search all.")

    selected_section = st.radio(
        label="Section",
        options=SECTIONS,
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("---")

    if metadata_by_id:
        st.markdown("### 📊 Index Status")
        total_chunks = len(metadata_by_id)
        dept_counts = {}
        source_files = set()
        for item in metadata_by_id.values():
            dept_counts[item["department"]] = dept_counts.get(item["department"], 0) + 1
            source_files.add(item["source_file"])

        st.caption(f"{total_chunks} chunks · {len(source_files)} document(s) loaded")
        for dept, count in sorted(dept_counts.items()):
            st.caption(f"• {dept}: {count} chunks")
    else:
        st.error(
            "No FAISS index found.\n\n"
            f"Expected files at:\n`{INDEX_PATH}`\n`{METADATA_PATH}`\n\n"
            "Run ingest.py first — this app never builds the index itself."
        )

    st.markdown("---")

    if groq_client is None:
        st.error(
            "GROQ_API_KEY not found in Streamlit secrets.\n\n"
            "Add it under Settings → Secrets:\n\n"
            'GROQ_API_KEY = "your-key-here"'
        )
    else:
        st.success("Groq API key loaded from secrets ✅")

    st.markdown("---")
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
def retrieve_chunks(query: str, section: str, top_k: int = TOP_K):
    """Embed the query, search FAISS, optionally filter by department."""
    if index is None or metadata_by_id is None:
        return []

    query_vector = embedder.encode([query]).astype("float32")

    fetch_k = OVER_FETCH_K if section != "All Sections" else top_k
    fetch_k = min(fetch_k, index.ntotal) if index.ntotal > 0 else 0
    if fetch_k == 0:
        return []

    distances, ids = index.search(query_vector, fetch_k)

    results = []
    for vec_id, dist in zip(ids[0], distances[0]):
        if vec_id == -1:
            continue
        item = metadata_by_id.get(int(vec_id))
        if item is None:
            continue
        if section != "All Sections" and item["department"] != section:
            continue
        results.append({**item, "distance": float(dist)})
        if len(results) >= top_k:
            break

    return results


# ---------------------------------------------------------------------------
# Answer generation via Groq
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are the Daraz Customer Support Operation Assistant.
You help Daraz customer support agents quickly find accurate answers about
Daraz's policies (returns, delivery, refunds, sellers, payments, and
customer support procedures).

Rules:
- Answer ONLY using the information given in the "Retrieved knowledge base
  context" below. Do not invent policy details that are not present there.
- If the context does not contain enough information to answer confidently,
  say so clearly and suggest the agent escalate or check the full policy
  document, rather than guessing.
- Be concise, professional, and practical — the reader is a support agent,
  not a customer. Give them the answer they can relay or act on directly.
- When useful, mention which section/department the answer comes from.
"""


def generate_answer(query: str, chunks: list):
    if groq_client is None:
        return "⚠️ Groq API key is not configured. Please add GROQ_API_KEY to Streamlit secrets."

    if not chunks:
        return (
            "I couldn't find anything relevant in the knowledge base for that "
            "question. Try rephrasing, or select a different section in the sidebar."
        )

    context_blocks = []
    for c in chunks:
        context_blocks.append(
            f"[Section: {c['department']} | Source: {c['source_file']} | Page: {c['page']}]\n{c['text']}"
        )
    context_text = "\n\n---\n\n".join(context_blocks)

    user_prompt = f"""Retrieved knowledge base context:

{context_text}

---

Agent's question: {query}

Answer the agent's question using only the context above."""

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=800,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Chat UI
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("📎 Sources used"):
                for s in message["sources"]:
                    st.markdown(
                        f'<span class="source-tag">{s["department"]}</span> '
                        f'{s["source_file"]} — page {s["page"]}',
                        unsafe_allow_html=True,
                    )

query = st.chat_input(
    f"Ask about {selected_section.lower() if selected_section != 'All Sections' else 'any policy'}..."
)

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            retrieved = retrieve_chunks(query, selected_section)
        with st.spinner("Generating answer..."):
            answer = generate_answer(query, retrieved)

        st.markdown(answer)
        if retrieved:
            with st.expander("📎 Sources used"):
                for s in retrieved:
                    st.markdown(
                        f'<span class="source-tag">{s["department"]}</span> '
                        f'{s["source_file"]} — page {s["page"]}',
                        unsafe_allow_html=True,
                    )

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": retrieved}
    )
