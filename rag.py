"""
RAG Studio — single-file Streamlit app
All modules (config, logger, styles, auth, retriever, rag_engine, main) merged.
Run: streamlit run app.py
"""

# ══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════════════════════════════════════

import os
import json
import time
import hashlib
import logging
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, TextLoader, CSVLoader,
)
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter, CharacterTextSplitter,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from sentence_transformers import CrossEncoder


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

DATA_DIR    = Path("rag_studio_data")
USERS_FILE  = DATA_DIR / "users.json"
HISTORY_DIR = DATA_DIR / "history"
LOGS_DIR    = DATA_DIR / "logs"

DATA_DIR.mkdir(exist_ok=True)
HISTORY_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

GROQ_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]

EMBED_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/paraphrase-MiniLM-L6-v2",
]

CHUNK_STRATEGIES = ["Recursive", "Character", "Sentence (Semantic)"]

DEFAULT_CHUNK_SIZE    = 500
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_TOP_K         = 10
DEFAULT_BM25_WEIGHT   = 0.4
DEFAULT_RERANK_TOP    = 4
MAX_FILES             = 10
MAX_HISTORY_DISK      = 50
MAX_HISTORY_PROFILE   = 10
RERANKER_MODEL        = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# ══════════════════════════════════════════════════════════════════════════════
# LOGGER
# ══════════════════════════════════════════════════════════════════════════════

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    logger.addHandler(console)

    log_file = LOGS_DIR / f"{datetime.now().strftime('%Y-%m-%d')}.log"
    file_h   = logging.FileHandler(log_file, encoding="utf-8")
    file_h.setLevel(logging.DEBUG)
    file_h.setFormatter(fmt)
    logger.addHandler(file_h)

    logger.propagate = False
    return logger

log = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# STYLES
# ══════════════════════════════════════════════════════════════════════════════

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
[data-testid="stSidebar"] { background: #0f1117; border-right: 1px solid #1e2130; }
[data-testid="stSidebar"] * { color: #c9d1d9 !important; }
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stTextInput label {
    color: #8b949e !important; font-size: 0.75rem !important;
    text-transform: uppercase; letter-spacing: 0.08em;
}
.main .block-container { background: #0d1117; padding-top: 2rem; }
body { background: #0d1117; }

.rag-title    { font-family: 'IBM Plex Mono', monospace; font-size: 1.9rem; font-weight: 600; color: #e6edf3; letter-spacing: -0.02em; }
.rag-subtitle { font-size: 0.85rem; color: #8b949e; margin-bottom: 1.5rem; }
.accent       { color: #58a6ff; }

.badge        { display: inline-block; padding: 2px 10px; border-radius: 20px; font-size: 0.72rem; font-family: 'IBM Plex Mono', monospace; font-weight: 600; letter-spacing: 0.05em; }
.badge-green  { background: #0d2818; color: #3fb950; border: 1px solid #238636; }
.badge-blue   { background: #0c1f3a; color: #58a6ff; border: 1px solid #1f6feb; }
.badge-gray   { background: #1c2128; color: #8b949e; border: 1px solid #30363d; }
.badge-orange { background: #2d1e0a; color: #f0883e; border: 1px solid #9e6a03; }
.badge-purple { background: #1a0f2e; color: #bc8cff; border: 1px solid #6e40c9; }

.qa-card  { background: #161b22; border: 1px solid #21262d; border-radius: 10px; padding: 1.1rem 1.3rem; margin-bottom: 1rem; }
.qa-q     { font-family: 'IBM Plex Mono', monospace; font-size: 0.82rem; color: #8b949e; margin-bottom: 0.4rem; text-transform: uppercase; letter-spacing: 0.06em; }
.qa-q span{ color: #58a6ff; font-weight: 600; }
.qa-a     { font-size: 0.95rem; color: #e6edf3; line-height: 1.65; margin-bottom: 0.6rem; white-space: pre-wrap; }
.qa-meta  { font-size: 0.72rem; color: #6e7681; font-family: 'IBM Plex Mono', monospace; }

.chunk-box   { background: #0d1117; border: 1px solid #1e2130; border-left: 3px solid #58a6ff; border-radius: 6px; padding: 0.75rem 1rem; margin-bottom: 0.6rem; font-size: 0.82rem; color: #8b949e; font-family: 'IBM Plex Mono', monospace; line-height: 1.6; white-space: pre-wrap; }
.chunk-label { font-size: 0.7rem; color: #58a6ff; font-family: 'IBM Plex Mono', monospace; margin-bottom: 0.3rem; text-transform: uppercase; letter-spacing: 0.08em; }
.rerank-score{ display: inline-block; background: #1a0f2e; color: #bc8cff; border: 1px solid #6e40c9; border-radius: 4px; font-size: 0.65rem; font-family: 'IBM Plex Mono', monospace; padding: 1px 6px; margin-left: 6px; }

.upload-hint { font-size: 0.78rem; color: #6e7681; margin-top: 0.3rem; }
.file-pill   { display: inline-block; background: #1c2128; border: 1px solid #30363d; border-radius: 5px; padding: 3px 9px; font-size: 0.72rem; color: #8b949e; font-family: 'IBM Plex Mono', monospace; margin: 2px; }
.info-box    { background: #0c1f3a; border: 1px solid #1f6feb; border-radius: 8px; padding: 0.8rem 1rem; font-size: 0.82rem; color: #8b949e; margin-bottom: 1rem; }
.divider     { border: none; border-top: 1px solid #21262d; margin: 1.2rem 0; }
.sidebar-section { font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; color: #58a6ff; text-transform: uppercase; letter-spacing: 0.12em; margin: 1.2rem 0 0.5rem 0; padding-bottom: 4px; border-bottom: 1px solid #1e2130; }

.stTextInput input       { background: #161b22 !important; border: 1px solid #30363d !important; color: #e6edf3 !important; border-radius: 8px !important; font-family: 'IBM Plex Sans', sans-serif !important; }
.stTextInput input:focus { border-color: #58a6ff !important; box-shadow: 0 0 0 3px rgba(88,166,255,0.12) !important; }
.stButton button         { background: #1f6feb !important; color: #ffffff !important; border: none !important; border-radius: 8px !important; font-family: 'IBM Plex Sans', sans-serif !important; font-weight: 600 !important; padding: 0.45rem 1.2rem !important; }
.stButton button:hover   { background: #388bfd !important; }

.profile-stat     { display: inline-block; background: #0d1117; border: 1px solid #21262d; border-radius: 8px; padding: 0.8rem 1.2rem; margin: 0.3rem; text-align: center; }
.profile-stat .val{ font-size: 1.6rem; font-weight: 600; font-family: 'IBM Plex Mono', monospace; color: #58a6ff; }
.profile-stat .lbl{ font-size: 0.65rem; color: #6e7681; text-transform: uppercase; letter-spacing: 0.08em; }

.hist-item  { background: #0d1117; border: 1px solid #21262d; border-left: 3px solid #1f6feb; border-radius: 6px; padding: 0.8rem 1rem; margin-bottom: 0.5rem; }
.hist-q     { font-size: 0.88rem; color: #c9d1d9; margin-bottom: 0.3rem; }
.hist-a     { font-size: 0.8rem; color: #6e7681; white-space: pre-wrap; }
.hist-meta  { font-size: 0.65rem; color: #444c56; font-family: 'IBM Plex Mono', monospace; margin-top: 0.3rem; }

.pipeline-bar  { display: flex; align-items: center; gap: 6px; font-size: 0.72rem; font-family: 'IBM Plex Mono', monospace; color: #6e7681; margin-bottom: 0.8rem; flex-wrap: wrap; }
.pipeline-step { background: #161b22; border: 1px solid #30363d; border-radius: 4px; padding: 2px 8px; }
.pipeline-step.active { border-color: #58a6ff; color: #58a6ff; }
.pipeline-arrow{ color: #30363d; }
</style>
"""


# ══════════════════════════════════════════════════════════════════════════════
# AUTH
# ══════════════════════════════════════════════════════════════════════════════

def hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

def load_users() -> dict:
    if not USERS_FILE.exists():
        return {}
    try:
        with open(USERS_FILE) as f:
            return json.load(f)
    except Exception as e:
        log.error(f"Failed to load users file: {e}")
        return {}

def save_users(users: dict) -> None:
    try:
        with open(USERS_FILE, "w") as f:
            json.dump(users, f, indent=2)
    except Exception as e:
        log.error(f"Failed to save users file: {e}")

def register_user(username: str, password: str, email: str = "") -> tuple[bool, str]:
    users = load_users()
    if username in users:
        return False, "Username already exists."
    users[username] = {
        "password":        hash_password(password),
        "email":           email,
        "created":         datetime.now().isoformat(),
        "total_questions": 0,
        "total_docs":      0,
    }
    save_users(users)
    log.info(f"New user registered: {username}")
    return True, "Account created!"

def login_user(username: str, password: str) -> tuple[bool, dict | str]:
    users = load_users()
    if username not in users:
        return False, "Username not found."
    if users[username]["password"] != hash_password(password):
        return False, "Incorrect password."
    log.info(f"User logged in: {username}")
    return True, users[username]

def update_user_stats(username: str, questions: int = 0, docs: int = 0) -> None:
    users = load_users()
    if username not in users:
        return
    users[username]["total_questions"] = users[username].get("total_questions", 0) + questions
    users[username]["total_docs"]      = users[username].get("total_docs", 0) + docs
    save_users(users)

def save_chat_entry(username: str, entry: dict) -> None:
    path    = HISTORY_DIR / f"{username}.json"
    history = []
    if path.exists():
        try:
            with open(path) as f:
                history = json.load(f)
        except Exception as e:
            log.error(f"Could not read history for {username}: {e}")
    history.insert(0, entry)
    history = history[:MAX_HISTORY_DISK]
    try:
        with open(path, "w") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        log.error(f"Could not write history for {username}: {e}")

def load_chat_history(username: str, limit: int = MAX_HISTORY_PROFILE) -> list:
    path = HISTORY_DIR / f"{username}.json"
    if not path.exists():
        return []
    try:
        with open(path) as f:
            return json.load(f)[:limit]
    except Exception as e:
        log.error(f"Could not load history for {username}: {e}")
        return []


# ══════════════════════════════════════════════════════════════════════════════
# RETRIEVER
# ══════════════════════════════════════════════════════════════════════════════

def load_file(uploaded_file) -> list:
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    try:
        loaders = {
            ".pdf":  lambda p: PyPDFLoader(p),
            ".docx": lambda p: Docx2txtLoader(p),
            ".doc":  lambda p: Docx2txtLoader(p),
            ".txt":  lambda p: TextLoader(p, encoding="utf-8"),
            ".csv":  lambda p: CSVLoader(file_path=p),
        }
        factory = loaders.get(suffix)
        if not factory:
            log.warning(f"Unsupported file type: {suffix}")
            return []
        docs = factory(tmp_path).load()
        for doc in docs:
            doc.metadata["source_filename"] = uploaded_file.name
        log.info(f"Loaded {len(docs)} pages from {uploaded_file.name}")
        return docs
    except Exception as e:
        log.error(f"Error loading {uploaded_file.name}: {e}")
        return []
    finally:
        os.unlink(tmp_path)

def get_splitter(strategy: str, size: int, overlap: int, embedding_model=None):
    if strategy == "Recursive":
        return RecursiveCharacterTextSplitter(
            chunk_size=size, chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    elif strategy == "Character":
        return CharacterTextSplitter(chunk_size=size, chunk_overlap=overlap, separator="\n")
    elif strategy == "Sentence (Semantic)":
        return SemanticChunker(embedding_model)

def build_indexes(docs, strategy, size, overlap, embed_model_name, use_parent_child):
    log.info(f"Building indexes | strategy={strategy} | parent_child={use_parent_child}")
    embedding = HuggingFaceEmbeddings(
        model_name=embed_model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    st.session_state.embedding_model = embedding

    if use_parent_child:
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=100)
        child_splitter  = RecursiveCharacterTextSplitter(chunk_size=300,  chunk_overlap=30)
        chunks = child_splitter.split_documents(parent_splitter.split_documents(docs))
    else:
        splitter = get_splitter(strategy, size, overlap, embedding)
        chunks   = splitter.split_documents(docs)

    vectorstore    = FAISS.from_documents(documents=chunks, embedding=embedding)
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 20

    log.info("FAISS + BM25 indexes ready")
    return vectorstore, bm25_retriever, chunks, len(chunks)

def dedupe_docs(docs: list) -> list:
    seen, out = set(), []
    for doc in docs:
        key = doc.page_content[:120]
        if key not in seen:
            seen.add(key)
            out.append(doc)
    return out

def rrf_merge(doc_lists: list[list], weights: list[float], k: int) -> list:
    assert len(doc_lists) == len(weights)
    scores, doc_map = {}, {}
    for docs, w in zip(doc_lists, weights):
        for rank, doc in enumerate(docs):
            key = doc.page_content[:120]
            scores[key]  = scores.get(key, 0) + w * (1 / (rank + 60))
            doc_map[key] = doc
    ranked = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [doc_map[key] for key in ranked[:k]]

def hybrid_retrieve(question: str, vectorstore, bm25_retriever, k: int, bm25_w: float) -> list:
    vec_docs         = vectorstore.similarity_search(question, k=k)
    bm25_retriever.k = k
    bm25_docs        = bm25_retriever.invoke(question)
    return rrf_merge([bm25_docs, vec_docs], [bm25_w, 1 - bm25_w], k)

def get_reranker():
    if st.session_state.reranker is None:
        with st.spinner("Loading reranker (first time only)…"):
            st.session_state.reranker = CrossEncoder(RERANKER_MODEL)
    return st.session_state.reranker

def rerank_docs(question: str, docs: list, reranker, top_n: int) -> tuple[list, list]:
    pairs  = [(question, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    top    = ranked[:top_n]
    return [doc for _, doc in top], [float(s) for s, _ in top]

def hyde_retrieve(question: str, vectorstore, llm, k: int) -> list:
    hyde_prompt = (
        f"Write a short factual paragraph that directly answers this question.\n"
        f"Be specific even if unsure — this is for search purposes only.\n"
        f"Question: {question}\nAnswer:"
    )
    hyp_doc = llm.invoke(hyde_prompt).content.strip()
    return vectorstore.similarity_search(hyp_doc, k=k)

def multi_query_retrieve(question: str, vectorstore, llm, k: int) -> list:
    mq_prompt = (
        f"Generate 3 different versions of this question to improve document retrieval.\n"
        f"Return ONLY the 3 questions, one per line, no numbering.\n"
        f"Original question: {question}"
    )
    variants_text = llm.invoke(mq_prompt).content.strip()
    queries = [q.strip() for q in variants_text.split("\n") if q.strip()][:3]
    queries.append(question)
    all_docs = []
    for q in queries:
        all_docs.extend(vectorstore.similarity_search(q, k=k))
    return dedupe_docs(all_docs)


# ══════════════════════════════════════════════════════════════════════════════
# RAG ENGINE
# ══════════════════════════════════════════════════════════════════════════════

PROMPT_TEMPLATE = """You are a knowledgeable assistant. Using ONLY the context provided below, answer the question clearly and accurately.
If the answer is not in the context, say: "I couldn't find relevant information in the uploaded documents."

Instructions:
- Be concise and direct.
- Use bullet points only when listing multiple items.
- Do not add introductions or conclusions.
- Do not make up information.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""


def rag_answer(
    question: str,
    vectorstore,
    bm25_retriever,
    llm,
    k: int,
    bm25_w: float,
    use_hybrid: bool,
    use_rerank: bool,
    rerank_top: int,
    use_hyde: bool,
    use_multiquery: bool,
) -> tuple[str, list, str, float]:
    log.info(f"RAG query | hybrid={use_hybrid} rerank={use_rerank} hyde={use_hyde} mq={use_multiquery}")

    candidate_lists = []
    methods         = []

    if use_hyde:
        hyde_docs = hyde_retrieve(question, vectorstore, llm, k)
        candidate_lists.append((hyde_docs, 0.35))
        methods.append("HyDE")

    if use_multiquery:
        mq_docs = multi_query_retrieve(question, vectorstore, llm, k)
        candidate_lists.append((mq_docs, 0.35))
        methods.append("Multi-Query")

    if use_hybrid and bm25_retriever:
        hybrid_docs = hybrid_retrieve(question, vectorstore, bm25_retriever, k, bm25_w)
        candidate_lists.append((hybrid_docs, 0.5))
        methods.append("Hybrid")
    else:
        vec_docs = vectorstore.similarity_search(question, k=k)
        candidate_lists.append((vec_docs, 1.0))
        methods.append("Vector")

    if len(candidate_lists) == 1:
        retrieved = candidate_lists[0][0]
    else:
        doc_lists = [dl for dl, _ in candidate_lists]
        weights   = [w  for _, w  in candidate_lists]
        total     = sum(weights)
        weights   = [w / total for w in weights]
        retrieved = rrf_merge(doc_lists, weights, k)

    rerank_scores = []
    if use_rerank and retrieved:
        reranker = get_reranker()
        top_docs, rerank_scores = rerank_docs(question, retrieved, reranker, rerank_top)
        methods.append("Reranked")
    else:
        top_docs = retrieved[:4]

    context  = "\n\n".join(doc.page_content for doc in top_docs)
    t0       = time.time()
    response = llm.invoke(PROMPT_TEMPLATE.format(context=context, question=question))
    latency  = round(time.time() - t0, 2)
    answer   = response.content.strip().replace("**", "")

    log.info(f"Answer generated | latency={latency}s | method={'+'.join(methods)}")

    chunk_info = [
        {
            "filename":     doc.metadata.get("source_filename", "Unknown"),
            "content":      doc.page_content,
            "rerank_score": round(rerank_scores[i], 3) if rerank_scores and i < len(rerank_scores) else None,
        }
        for i, doc in enumerate(top_docs[:3])
    ]

    return answer, chunk_info, " + ".join(methods), latency


# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT APP
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="RAG Studio",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CSS, unsafe_allow_html=True)

DEFAULTS = {
    "logged_in":       False,
    "username":        None,
    "user_data":       {},
    "page":            "login",
    "vectorstore":     None,
    "bm25_retriever":  None,
    "all_chunks":      [],
    "chat_history":    [],
    "processed_files": [],
    "total_chunks":    0,
    "embedding_model": None,
    "reranker":        None,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Auth pages ────────────────────────────────────────────────────────────────

def render_login():
    _, col, _ = st.columns([1, 1.4, 1])
    with col:
        st.markdown('<div class="rag-title" style="text-align:center;"> RAG Studio</div>', unsafe_allow_html=True)
        st.markdown('<div class="rag-subtitle" style="text-align:center;">Sign in to continue</div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("### Sign In")
            username = st.text_input("Username", key="li_user")
            password = st.text_input("Password", type="password", key="li_pass")
            c1, c2   = st.columns(2)
            with c1:
                if st.button("Login", use_container_width=True):
                    if username and password:
                        ok, result = login_user(username, password)
                        if ok:
                            st.session_state.logged_in = True
                            st.session_state.username  = username
                            st.session_state.user_data = result
                            st.session_state.page      = "app"
                            st.rerun()
                        else:
                            st.error(result)
                    else:
                        st.warning("Fill in all fields.")
            with c2:
                if st.button("Create Account", use_container_width=True):
                    st.session_state.page = "signup"
                    st.rerun()


def render_signup():
    _, col, _ = st.columns([1, 1.4, 1])
    with col:
        with st.container(border=True):
            st.markdown("### Create Account")
            username = st.text_input("Username", key="su_user")
            email    = st.text_input("Email (optional)", key="su_email")
            password = st.text_input("Password", type="password", key="su_pass")
            confirm  = st.text_input("Confirm Password", type="password", key="su_conf")
            c1, c2   = st.columns(2)
            with c1:
                if st.button("Sign Up", use_container_width=True):
                    if not username or not password:
                        st.warning("Username and password required.")
                    elif password != confirm:
                        st.error("Passwords do not match.")
                    elif len(password) < 6:
                        st.error("Password must be at least 6 characters.")
                    else:
                        ok, msg = register_user(username, password, email)
                        if ok:
                            st.success(msg + " Please login.")
                            st.session_state.page = "login"
                            st.rerun()
                        else:
                            st.error(msg)
            with c2:
                if st.button("Back to Login", use_container_width=True):
                    st.session_state.page = "login"
                    st.rerun()


# ── Profile page ──────────────────────────────────────────────────────────────

def render_profile():
    udata   = load_users().get(st.session_state.username, {})
    history = load_chat_history(st.session_state.username)
    joined  = udata.get("created", "")[:10]

    st.markdown('<div class="rag-title"> Profile</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="rag-subtitle">@{st.session_state.username}</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div style="display:flex;flex-wrap:wrap;gap:10px;margin-bottom:1.5rem;">
        <div class="profile-stat"><div class="val">{udata.get('total_questions', 0)}</div><div class="lbl">Questions Asked</div></div>
        <div class="profile-stat"><div class="val">{udata.get('total_docs', 0)}</div><div class="lbl">Docs Uploaded</div></div>
        <div class="profile-stat"><div class="val">{len(history)}</div><div class="lbl">Saved Chats</div></div>
        <div class="profile-stat"><div class="val" style="font-size:1rem;">{joined}</div><div class="lbl">Joined</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div style="font-family:\'IBM Plex Mono\',monospace;font-size:0.75rem;color:#58a6ff;margin-bottom:1rem;">LAST 10 CONVERSATIONS</div>', unsafe_allow_html=True)

    if not history:
        st.markdown('<div style="color:#6e7681;padding:1rem 0;">No conversations yet.</div>', unsafe_allow_html=True)
    else:
        for item in history:
            ts    = item.get("timestamp", "")[:16].replace("T", " ")
            short = item.get("answer", "")[:200]
            short += "…" if len(item.get("answer", "")) > 200 else ""
            st.markdown(
                f'<div class="hist-item">'
                f'<div class="hist-q">Q: {item.get("question", "")}</div>'
                f'<div class="hist-a">{short}</div>'
                f'<div class="hist-meta">{ts} · {item.get("model", "—")} · {item.get("method", "—")}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    if st.button("← Back to App"):
        st.session_state.page = "app"
        st.rerun()


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown('<div class="rag-title" style="font-size:1.1rem;"> RAG Studio</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div style="font-size:0.75rem;color:#6e7681;margin-bottom:0.5rem;">'
            f'Logged in as <span style="color:#58a6ff;">@{st.session_state.username}</span></div>',
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns(2)
        with c1:
            if st.button(" Profile", use_container_width=True):
                st.session_state.page = "profile"
                st.rerun()
        with c2:
            if st.button("Logout", use_container_width=True):
                for k in list(DEFAULTS.keys()):
                    st.session_state[k] = DEFAULTS[k]
                st.session_state.page = "login"
                st.rerun()

        st.markdown('<div class="sidebar-section"> API</div>', unsafe_allow_html=True)
        groq_api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_…", key="sb_groq")
        groq_model   = st.selectbox("Model", GROQ_MODELS, index=0)
        temperature  = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)

        st.markdown('<div class="sidebar-section" Chunking</div>', unsafe_allow_html=True)
        chunk_strategy = st.selectbox("Strategy", CHUNK_STRATEGIES, index=0)
        if chunk_strategy != "Sentence (Semantic)":
            chunk_size    = st.slider("Chunk Size",    100, 2000, DEFAULT_CHUNK_SIZE,    50)
            chunk_overlap = st.slider("Chunk Overlap",   0,  500, DEFAULT_CHUNK_OVERLAP, 10)
        else:
            st.markdown('<div class="info-box">Semantic chunking auto-sizes chunks.</div>', unsafe_allow_html=True)
            chunk_size, chunk_overlap = DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

        st.markdown('<div class="sidebar-section"> Embedding</div>', unsafe_allow_html=True)
        embed_model_name = st.selectbox("HuggingFace Model", EMBED_MODELS, index=0)

        st.markdown('<div class="sidebar-section"> Level 1 — Retrieval</div>', unsafe_allow_html=True)
        top_k      = st.slider("Top K pool", 5, 30, DEFAULT_TOP_K, 1)
        use_hybrid = st.toggle(" Hybrid Search (BM25 + Vector)", value=True)
        bm25_w     = st.slider("BM25 weight", 0.1, 0.9, DEFAULT_BM25_WEIGHT, 0.05, disabled=not use_hybrid)
        use_rerank = st.toggle(" Cross-Encoder Reranker", value=True)
        rerank_top = st.slider("Chunks after rerank", 1, 10, DEFAULT_RERANK_TOP, 1, disabled=not use_rerank)

        st.markdown('<div class="sidebar-section"> Level 2 — Advanced</div>', unsafe_allow_html=True)
        use_hyde         = st.toggle(" HyDE",              value=False, help="Hypothetical Document Embeddings")
        use_multiquery   = st.toggle(" Multi-Query",        value=False, help="Generate 3 query variants")
        use_parent_child = st.toggle(" Parent-Child Chunking", value=False)

        st.markdown("---")
        if st.button(" Clear Session Chat"):
            st.session_state.chat_history = []
            st.rerun()

    return dict(
        groq_api_key=groq_api_key, groq_model=groq_model, temperature=temperature,
        chunk_strategy=chunk_strategy, chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        embed_model_name=embed_model_name, top_k=top_k,
        use_hybrid=use_hybrid, bm25_w=bm25_w,
        use_rerank=use_rerank, rerank_top=rerank_top,
        use_hyde=use_hyde, use_multiquery=use_multiquery,
        use_parent_child=use_parent_child,
    )


# ── Main app page ─────────────────────────────────────────────────────────────

def render_app():
    cfg = render_sidebar()

    st.markdown('<div class="rag-title"> RAG <span class="accent">Studio</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="rag-subtitle">Hybrid Search · HyDE · Multi-Query · Parent-Child Chunking</div>', unsafe_allow_html=True)

    steps = []
    if cfg["use_hyde"]:         steps.append(" HyDE")
    if cfg["use_multiquery"]:   steps.append(" Multi-Query")
    if cfg["use_hybrid"]:       steps.append(" Hybrid")
    if cfg["use_rerank"]:       steps.append(" Reranker")
    if cfg["use_parent_child"]: steps.append(" Parent-Child")
    if not steps:               steps = ["Vector"]
    bar = " <span class='pipeline-arrow'>→</span> ".join(
        f'<span class="pipeline-step active">{s}</span>' for s in steps
    )
    st.markdown(
        f'<div class="pipeline-bar">{bar} <span class="pipeline-arrow">→</span> '
        f'<span class="pipeline-step">LLM</span></div>',
        unsafe_allow_html=True,
    )

    # File upload
    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded_files = st.file_uploader(
            "Upload", type=["pdf", "docx", "txt", "csv"],
            accept_multiple_files=True, label_visibility="collapsed",
        )
        st.markdown('<div class="upload-hint"> PDF, DOCX, TXT, CSV — multiple files allowed</div>', unsafe_allow_html=True)
    with col2:
        process_btn = st.button(" Process Files", use_container_width=True)
        if st.session_state.vectorstore:
            st.markdown(
                f'<div style="margin-top:6px"><span class="badge badge-green"> READY</span> '
                f'<span class="badge badge-blue">{st.session_state.total_chunks} chunks</span></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<div style="margin-top:6px"><span class="badge badge-gray">NOT PROCESSED</span></div>', unsafe_allow_html=True)

    if uploaded_files:
        pills = "".join(f'<span class="file-pill"> {f.name}</span>' for f in uploaded_files)
        st.markdown(f'<div style="margin-top:0.5rem;">{pills}</div>', unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Process documents
    if process_btn:
        if not uploaded_files:
            st.warning("Please upload at least one file.")
        elif len(uploaded_files) > MAX_FILES:
            st.error(f"Maximum {MAX_FILES} files allowed.")
        elif not cfg["groq_api_key"]:
            st.warning("Add your Groq API key in the sidebar.")
        else:
            with st.spinner("Processing documents…"):
                all_docs, failed = [], []
                for uf in uploaded_files:
                    docs = load_file(uf)
                    if docs:
                        all_docs.extend(docs)
                    else:
                        failed.append(uf.name)

                if all_docs:
                    try:
                        vs, bm25_ret, chunks, n = build_indexes(
                            all_docs,
                            cfg["chunk_strategy"], cfg["chunk_size"], cfg["chunk_overlap"],
                            cfg["embed_model_name"], cfg["use_parent_child"],
                        )
                        st.session_state.vectorstore     = vs
                        st.session_state.bm25_retriever  = bm25_ret
                        st.session_state.all_chunks      = chunks
                        st.session_state.total_chunks    = n
                        st.session_state.processed_files = [f.name for f in uploaded_files]
                        update_user_stats(st.session_state.username, docs=len(uploaded_files))
                        st.success(f" {len(uploaded_files)} file(s) → {n} chunks ready!")
                        log.info(f"Processed {len(uploaded_files)} files, {n} chunks")
                    except Exception as e:
                        log.error(f"build_indexes error: {e}")
                        st.error(f"Error building index: {e}")
                else:
                    st.error("Could not load any documents.")
                if failed:
                    st.warning(f"Failed to load: {', '.join(failed)}")
            st.rerun()

    # Prompt user to get started
    if not st.session_state.vectorstore:
        st.markdown(
            '<div class="info-box"> <strong>Getting started:</strong> Upload documents, '
            'add your Groq API key in the sidebar, then click <strong>Process Files</strong>.</div>',
            unsafe_allow_html=True,
        )
        return

    # Status row
    files_str     = " · ".join(st.session_state.processed_files)
    active_badges = ""
    if cfg["use_hybrid"]:       active_badges += '<span class="badge badge-blue"> Hybrid</span> '
    if cfg["use_rerank"]:       active_badges += '<span class="badge badge-purple"> Reranker</span> '
    if cfg["use_hyde"]:         active_badges += '<span class="badge badge-purple"> HyDE</span> '
    if cfg["use_multiquery"]:   active_badges += '<span class="badge badge-purple"> Multi-Q</span> '
    if cfg["use_parent_child"]: active_badges += '<span class="badge badge-orange"> P-C</span> '

    st.markdown(
        f'<div style="display:flex;gap:8px;align-items:center;margin-bottom:1rem;flex-wrap:wrap;">'
        f'<span class="badge badge-green"> READY</span>'
        f'<span class="badge badge-blue">FAISS+BM25 · {st.session_state.total_chunks} chunks</span>'
        f'<span class="badge badge-orange">{cfg["chunk_strategy"]}</span>'
        f'{active_badges}'
        f'<span style="font-size:0.72rem;color:#6e7681;">{files_str}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Question input
    col_q, col_btn = st.columns([5, 1])
    with col_q:
        user_question = st.text_input(
            "Ask", placeholder="What does the document say about…?",
            label_visibility="collapsed", key="question_input",
        )
    with col_btn:
        ask_btn = st.button("Ask →", use_container_width=True)

    if ask_btn and user_question.strip():
        if not cfg["groq_api_key"]:
            st.warning("Add your Groq API key.")
        else:
            with st.spinner("Thinking…"):
                try:
                    os.environ["GROQ_API_KEY"] = cfg["groq_api_key"]
                    llm = ChatGroq(
                        model=cfg["groq_model"],
                        temperature=cfg["temperature"],
                        api_key=cfg["groq_api_key"],
                    )
                    answer, chunk_info, method, latency = rag_answer(
                        user_question,
                        st.session_state.vectorstore,
                        st.session_state.bm25_retriever,
                        llm,
                        cfg["top_k"], cfg["bm25_w"],
                        cfg["use_hybrid"], cfg["use_rerank"], cfg["rerank_top"],
                        cfg["use_hyde"], cfg["use_multiquery"],
                    )
                    entry = {
                        "question":  user_question,
                        "answer":    answer,
                        "chunks":    chunk_info,
                        "model":     cfg["groq_model"],
                        "method":    method,
                        "latency":   latency,
                        "timestamp": datetime.now().isoformat(),
                    }
                    st.session_state.chat_history.insert(0, entry)
                    save_chat_entry(st.session_state.username, entry)
                    update_user_stats(st.session_state.username, questions=1)
                except Exception as e:
                    log.error(f"rag_answer error: {e}")
                    st.error(f"Error: {e}")
            st.rerun()

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Chat history
    if st.session_state.chat_history:
        n_total = len(st.session_state.chat_history)
        st.markdown(
            f'<div style="font-size:0.75rem;color:#6e7681;margin-bottom:0.8rem;'
            f'font-family:\'IBM Plex Mono\',monospace;">SESSION HISTORY · {n_total} exchange(s)</div>',
            unsafe_allow_html=True,
        )
        for i, item in enumerate(st.session_state.chat_history):
            q_num  = n_total - i
            answer = item.get("answer", "")
            model  = item.get("model", "—")
            method = item.get("method", "—")
            lat    = item.get("latency", "—")
            nchunk = len(item.get("chunks", []))

            st.markdown(
                f'<div class="qa-card">'
                f'<div class="qa-q"><span>Q{q_num}</span> · {item["question"]}</div>'
                f'<div class="qa-a">{answer}</div>'
                f'<div class="qa-meta">model: {model} · pipeline: {method} · latency: {lat}s · {nchunk} chunks</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            with st.expander(f" Source chunks — Q{q_num}"):
                for j, chunk in enumerate(item.get("chunks", [])):
                    rs        = chunk.get("rerank_score")
                    score_tag = f'<span class="rerank-score">rerank: {rs}</span>' if rs is not None else ""
                    preview   = chunk["content"][:600] + ("…" if len(chunk["content"]) > 600 else "")
                    st.markdown(
                        f'<div class="chunk-label">Chunk {j+1} · {chunk["filename"]} {score_tag}</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(f'<div class="chunk-box">{preview}</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div style="text-align:center;padding:2rem;color:#6e7681;font-size:0.85rem;">'
            'Ask your first question above ⬆</div>',
            unsafe_allow_html=True,
        )


# ── Router ────────────────────────────────────────────────────────────────────

if not st.session_state.logged_in:
    if st.session_state.page == "signup":
        render_signup()
    else:
        render_login()
else:
    if st.session_state.page == "profile":
        render_profile()
    else:
        render_app()
