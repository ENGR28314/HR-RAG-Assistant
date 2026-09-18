import html
import re
from typing import List, Dict

import faiss
import fitz
import numpy as np
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="👩‍💼",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# APPLICATION CONSTANTS
# ============================================================

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-20b"

CHUNK_SIZE = 180
CHUNK_OVERLAP = 40

TOP_K = 5
SIMILARITY_THRESHOLD = 0.30


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 750;
        margin-bottom: 4px;
    }

    .subtitle {
        font-size: 18px;
        color: #6b7280;
        margin-bottom: 25px;
    }

    .metric-card {
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        background-color: #f9fafb;
    }

    .source-card {
        padding: 14px;
        border-radius: 10px;
        border: 1px solid #e5e7eb;
        margin-bottom: 10px;
        background-color: #ffffff;
    }

    .answer-header {
        font-size: 22px;
        font-weight: 700;
    }

    .small-text {
        font-size: 13px;
        color: #6b7280;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# INITIALIZE SESSION STATE
# ============================================================

defaults = {
    "messages": [],
    "chunks": [],
    "index": None,
    "document_name": None,
    "document_pages": 0,
    "document_text_length": 0,
    "processed_file_id": None,
    "last_retrieval": [],
}

for key, value in defaults.items():

    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">👩‍💼 HR Policy Assistant</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
    Ask questions about an uploaded HR Policy PDF using
    Retrieval-Augmented Generation (RAG).
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_groq_api_key() -> str:
    """
    Read the Groq API key from Streamlit Secrets.

    Expected secret:

    GROQ_API_KEY = "your-key"
    """

    try:
        key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        key = ""

    return str(key).strip()


# ------------------------------------------------------------
# Text cleaning
# ------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Clean extracted PDF text while preserving readable content.
    """

    text = text.replace("\x00", " ")
    text = text.replace("\r", "\n")

    # Remove excessive spaces.
    text = re.sub(r"[ \t]+", " ", text)

    # Reduce excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ------------------------------------------------------------
# PDF extraction
# ------------------------------------------------------------

def extract_pdf_pages(pdf_bytes: bytes) -> List[Dict]:
    """
    Extract text page-by-page so retrieved chunks can
    reference the original PDF page.
    """

    pages = []

    document = fitz.open(
        stream=pdf_bytes,
        filetype="pdf",
    )

    try:

        for page_number, page in enumerate(
            document,
            start=1,
        ):

            text = clean_text(
                page.get_text("text")
            )

            if text:

                pages.append(
                    {
                        "page": page_number,
                        "text": text,
                    }
                )

    finally:

        document.close()

    return pages


# ------------------------------------------------------------
# Chunking
# ------------------------------------------------------------

def chunk_page_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """
    Create word-based chunks with overlap.
    """

    words = text.split()

    if not words:
        return []

    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 4)

    chunks = []

    start = 0
    step = chunk_size - overlap

    while start < len(words):

        end = min(
            start + chunk_size,
            len(words),
        )

        chunk = " ".join(
            words[start:end]
        ).strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(words):
            break

        start += step

    return chunks


def create_document_chunks(
    pages: List[Dict],
) -> List[Dict]:
    """
    Create searchable chunks while retaining page metadata.
    """

    chunks = []

    chunk_id = 0

    for page_data in pages:

        page_number = page_data["page"]
        page_text = page_data["text"]

        page_chunks = chunk_page_text(
            page_text
        )

        for chunk in page_chunks:

            chunks.append(
                {
                    "id": chunk_id,
                    "page": page_number,
                    "text": chunk,
                }
            )

            chunk_id += 1

    return chunks


# ------------------------------------------------------------
# Embedding generation
# ------------------------------------------------------------

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        EMBEDDING_MODEL
    )


def embed_texts(
    texts: List[str],
) -> np.ndarray:

    model = load_embedding_model()

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return np.asarray(
        embeddings,
        dtype="float32",
    )


# ------------------------------------------------------------
# FAISS index
# ------------------------------------------------------------

def build_faiss_index(
    chunks: List[Dict],
):

    texts = [
        item["text"]
        for item in chunks
    ]

    embeddings = embed_texts(
        texts
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    return index


# ------------------------------------------------------------
# Retrieve chunks
# ------------------------------------------------------------

def retrieve_relevant_chunks(
    question: str,
    index,
    chunks: List[Dict],
    top_k: int = TOP_K,
    threshold: float = SIMILARITY_THRESHOLD,
) -> List[Dict]:

    if index is None or not chunks:
        return []

    query_embedding = embed_texts(
        [question]
    )

    k = min(
        top_k,
        len(chunks),
    )

    scores, indices = index.search(
        query_embedding,
        k,
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0],
    ):

        if idx < 0:
            continue

        similarity = float(score)

        item = chunks[idx].copy()

        item["similarity"] = similarity

        if similarity >= threshold:
            results.append(item)

    return results


# ------------------------------------------------------------
# Build RAG context
# ------------------------------------------------------------

def build_context(
    retrieved_chunks: List[Dict],
) -> str:

    sections = []

    for i, item in enumerate(
        retrieved_chunks,
        start=1,
    ):

        sections.append(
            f"""
[Retrieved Policy Section {i}]
PDF Page: {item["page"]}
Similarity: {item["similarity"]:.3f}

{item["text"]}
""".strip()
        )

    return "\n\n".join(
        sections
    )


# ------------------------------------------------------------
# Groq response
# ------------------------------------------------------------

def generate_answer(
    question: str,
    retrieved_chunks: List[Dict],
    api_key: str,
) -> str:

    if not retrieved_chunks:

        return (
            "I could not find sufficiently relevant "
            "information in the uploaded HR policy to "
            "answer this question. Please check the "
            "policy document or ask a question related "
            "to its contents."
        )

    client = Groq(
        api_key=api_key
    )

    context = build_context(
        retrieved_chunks
    )

    system_prompt = """
You are an HR Policy Assistant.

You answer questions about an organization's HR policy.

IMPORTANT RULES:

1. Use ONLY the supplied HR POLICY CONTEXT.
2. Do not use general knowledge to fill missing policy information.
3. Do not invent rules, benefits, salaries, leave allowances,
   working hours, disciplinary procedures, or other HR policies.
4. If the supplied context does not contain enough information,
   say that the information is not available in the uploaded
   HR policy.
5. Keep answers professional, clear, and concise.
6. When possible, mention the relevant PDF page number.
7. If the policy gives conditions or exceptions, include them.
8. Do not claim that a policy exists unless it is supported
   by the supplied context.
9. Treat the uploaded policy as the authoritative source for
   this question.
"""

    user_prompt = f"""
HR POLICY CONTEXT:

{context}

EMPLOYEE QUESTION:

{question}

Provide the answer using only the HR POLICY CONTEXT.
"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=0.1,
        max_tokens=800,
    )

    answer = response.choices[0].message.content

    return answer.strip()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Assistant Settings")

    api_key = get_groq_api_key()

    if api_key:

        st.success(
            "Groq API key loaded from Streamlit Secrets."
        )

    else:

        st.error(
            "GROQ_API_KEY is not configured."
        )

        st.info(
            "Add GROQ_API_KEY in your Streamlit Cloud "
            "Secrets settings."
        )

    st.markdown("---")

    st.subheader("🤖 AI Model")

    st.code(
        GROQ_MODEL
    )

    st.subheader("🧠 Embedding Model")

    st.code(
        EMBEDDING_MODEL
    )

    st.markdown("---")

    st.subheader("🔎 Retrieval")

    st.write(
        f"Top results: {TOP_K}"
    )

    st.write(
        f"Similarity threshold: {SIMILARITY_THRESHOLD}"
    )

    st.markdown("---")

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True,
    ):

        st.session_state.messages = []
        st.session_state.last_retrieval = []

        st.rerun()

    if st.button(
        "🔄 Reset Document",
        use_container_width=True,
    ):

        st.session_state.chunks = []
        st.session_state.index = None
        st.session_state.document_name = None
        st.session_state.document_pages = 0
        st.session_state.document_text_length = 0
        st.session_state.processed_file_id = None
        st.session_state.messages = []
        st.session_state.last_retrieval = []

        st.rerun()

    st.markdown("---")

    st.caption(
        "RAG architecture: "
        "PDF → embeddings → FAISS → retrieval → Groq"
    )


# ============================================================
# PDF UPLOAD
# ============================================================

st.header("📄 1. Upload HR Policy")

uploaded_file = st.file_uploader(
    "Upload an HR Policy PDF",
    type=["pdf"],
    help="Upload a text-based HR policy PDF.",
)


# ============================================================
# PROCESS PDF
# ============================================================

if uploaded_file is not None:

    # Streamlit UploadedFile provides a stable name and size.
    current_file_id = (
        f"{uploaded_file.name}-"
        f"{uploaded_file.size}"
    )

    if (
        st.session_state.processed_file_id
        != current_file_id
    ):

        with st.spinner(
            "Reading and indexing HR policy..."
        ):

            pdf_bytes = uploaded_file.getvalue()

            pages = extract_pdf_pages(
                pdf_bytes
            )

            if not pages:

                st.error(
                    "No readable text was found in this PDF. "
                    "If the document is scanned, OCR may be required."
                )

                st.stop()

            chunks = create_document_chunks(
                pages
            )

            if not chunks:

                st.error(
                    "No searchable text chunks could be created."
                )

                st.stop()

            index = build_faiss_index(
                chunks
            )

            st.session_state.index = index
            st.session_state.chunks = chunks

            st.session_state.document_name = (
                uploaded_file.name
            )

            st.session_state.document_pages = (
                len(pages)
            )

            st.session_state.document_text_length = sum(
                len(page["text"])
                for page in pages
            )

            st.session_state.processed_file_id = (
                current_file_id
            )

            st.session_state.messages = []
            st.session_state.last_retrieval = []

        st.success(
            "HR policy indexed successfully."
        )


# ============================================================
# DOCUMENT INFORMATION
# ============================================================

if st.session_state.document_name:

    st.header("📊 2. Document Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Document",
            st.session_state.document_name,
        )

    with col2:

        st.metric(
            "Pages",
            st.session_state.document_pages,
        )

    with col3:

        st.metric(
            "Searchable Chunks",
            len(st.session_state.chunks),
        )


# ============================================================
# EXAMPLE QUESTIONS
# ============================================================

st.header("💡 3. Example Questions")

example_questions = [
    "How many annual leave days are employees entitled to?",
    "What is the resignation notice period?",
    "What are the standard working hours?",
    "What is the sick leave policy?",
    "Can employees work remotely?",
    "What is the company's maternity leave policy?",
]

example_cols = st.columns(2)

for i, example in enumerate(
    example_questions
):

    with example_cols[i % 2]:

        if st.button(
            example,
            key=f"example_{i}",
            use_container_width=True,
        ):

            st.session_state.pending_question = (
                example
            )


# ============================================================
# QUESTION INPUT
# ============================================================

st.header("💬 4. Ask the HR Policy Assistant")

pending_question = st.session_state.pop(
    "pending_question",
    "",
)

question = st.text_input(
    "Your question",
    value=pending_question,
    placeholder=(
        "Example: What is the resignation notice period?"
    ),
)


ask_button = st.button(
    "🔎 Ask Question",
    type="primary",
    use_container_width=True,
)


# ============================================================
# QUESTION PROCESSING
# ============================================================

if ask_button:

    if not api_key:

        st.error(
            "GROQ_API_KEY is not configured. "
            "Please add it to Streamlit Cloud Secrets."
        )

        st.stop()

    if st.session_state.index is None:

        st.warning(
            "Please upload an HR Policy PDF first."
        )

        st.stop()

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

        st.stop()

    clean_question = question.strip()

    with st.spinner(
        "Searching the HR policy..."
    ):

        retrieved_chunks = retrieve_relevant_chunks(
            clean_question,
            st.session_state.index,
            st.session_state.chunks,
        )

    st.session_state.last_retrieval = (
        retrieved_chunks
    )

    with st.spinner(
        "Generating policy-based answer..."
    ):

        try:

            answer = generate_answer(
                clean_question,
                retrieved_chunks,
                api_key,
            )

        except Exception as error:

            st.error(
                "The Groq request failed."
            )

            st.exception(
                error
            )

            st.stop()

    st.session_state.messages.append(
        {
            "question": clean_question,
            "answer": answer,
            "sources": retrieved_chunks,
        }
    )


# ============================================================
# CHAT HISTORY
# ============================================================

if st.session_state.messages:

    st.header("🗨️ Conversation")

    for message in reversed(
        st.session_state.messages
    ):

        with st.chat_message(
            "user"
        ):

            st.write(
                message["question"]
            )

        with st.chat_message(
            "assistant"
        ):

            st.write(
                message["answer"]
            )

            sources = message.get(
                "sources",
                [],
            )

            if sources:

                with st.expander(
                    "📚 View retrieved policy sources"
                ):

                    for source_number, source in enumerate(
                        sources,
                        start=1,
                    ):

                        st.markdown(
                            f"""
                            <div class="source-card">
                            <strong>
                            Source {source_number}
                            — PDF Page {source["page"]}
                            </strong>
                            <br>
                            <span class="small-text">
                            Similarity:
                            {source["similarity"]:.3f}
                            </span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        st.write(
                            source["text"]
                        )

                        st.markdown("---")


# ============================================================
# RAG DETAILS
# ============================================================

if st.session_state.last_retrieval:

    st.header("🔬 Latest Retrieval Details")

    st.caption(
        "These are the policy sections retrieved before "
        "the latest answer was generated."
    )

    for i, item in enumerate(
        st.session_state.last_retrieval,
        start=1,
    ):

        st.markdown(
            f"**Result {i} — Page {item['page']} "
            f"— Similarity {item['similarity']:.3f}**"
        )

        st.write(
            item["text"]
        )

        st.markdown("---")


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "HR Policy Assistant • "
    "RAG + FAISS + Sentence Transformers + PyMuPDF + Groq"
)
