import json
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Daraz Customer Support Operations Assistant",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CONFIGURATION
# ============================================================

INDEX_DIR = Path("faiss_index")

FAISS_INDEX_PATH = INDEX_DIR / "index.faiss"
METADATA_PATH = INDEX_DIR / "metadata.json"
CONFIG_PATH = INDEX_DIR / "config.json"

MODEL_NAME = "openai/gpt-oss-120b"

DEFAULT_EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

TOP_K = 5


# ============================================================
# DARAZ-STYLE CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background-color: #f7f8fa;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e6e6e6;
    }

    /* Main content width */
    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Header */
    .daraz-header {
        background: linear-gradient(
            135deg,
            #f85606 0%,
            #ff7a2f 100%
        );

        padding: 28px 32px;
        border-radius: 18px;
        color: white;
        margin-bottom: 25px;

        box-shadow:
            0 8px 25px rgba(248, 86, 6, 0.18);
    }

    .daraz-header h1 {
        margin: 0;
        font-size: 32px;
        font-weight: 750;
    }

    .daraz-header p {
        margin-top: 8px;
        margin-bottom: 0;
        font-size: 15px;
        opacity: 0.95;
    }

    /* Section cards */
    .info-card {
        background: white;
        border: 1px solid #eeeeee;
        border-radius: 15px;
        padding: 18px;
        margin-bottom: 15px;

        box-shadow:
            0 3px 12px rgba(0, 0, 0, 0.04);
    }

    /* Source card */
    .source-card {
        background: #fff8f3;
        border-left: 4px solid #f85606;
        padding: 12px 15px;
        margin-top: 8px;
        border-radius: 8px;
        font-size: 13px;
    }

    /* Small label */
    .small-label {
        font-size: 12px;
        color: #777777;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
    }

    /* Status */
    .status-online {
        color: #1f9d55;
        font-weight: 600;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 9px;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="daraz-header">
        <h1>🛍️ Daraz Customer Support Operations Assistant</h1>
        <p>
            AI-powered support assistant grounded in the Daraz
            Knowledge Base.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPER: LOAD CONFIG
# ============================================================

@st.cache_resource(show_spinner=False)
def load_config():

    if CONFIG_PATH.exists():

        with open(
            CONFIG_PATH,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    return {
        "embedding_model":
            DEFAULT_EMBEDDING_MODEL
    }


# ============================================================
# LOAD FAISS INDEX
# ============================================================

@st.cache_resource(show_spinner=False)
def load_faiss_index():

    if not FAISS_INDEX_PATH.exists():

        raise FileNotFoundError(
            f"FAISS index not found: "
            f"{FAISS_INDEX_PATH}"
        )

    return faiss.read_index(
        str(FAISS_INDEX_PATH)
    )


# ============================================================
# LOAD METADATA
# ============================================================

@st.cache_data(show_spinner=False)
def load_metadata():

    if not METADATA_PATH.exists():

        raise FileNotFoundError(
            f"Metadata file not found: "
            f"{METADATA_PATH}"
        )

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def load_embedding_model(model_name):

    return SentenceTransformer(
        model_name
    )


# ============================================================
# LOAD GROQ CLIENT
# ============================================================

@st.cache_resource(show_spinner=False)
def load_groq_client():

    if "GROQ_API_KEY" not in st.secrets:

        raise RuntimeError(
            "GROQ_API_KEY is missing from "
            "Streamlit Secrets."
        )

    api_key = st.secrets[
        "GROQ_API_KEY"
    ]

    return Groq(
        api_key=api_key
    )


# ============================================================
# LOAD RAG COMPONENTS
# ============================================================

try:

    config = load_config()

    embedding_model_name = config.get(
        "embedding_model",
        DEFAULT_EMBEDDING_MODEL
    )

    with st.spinner(
        "Loading Daraz knowledge base..."
    ):

        index = load_faiss_index()

        metadata = load_metadata()

        embedding_model = (
            load_embedding_model(
                embedding_model_name
            )
        )

        groq_client = load_groq_client()

except Exception as e:

    st.error(
        "Unable to initialize the application."
    )

    st.code(
        str(e)
    )

    st.stop()


# ============================================================
# DEPARTMENTS
# ============================================================

DEPARTMENTS = {
    "All Knowledge Base": None,

    "Returns": "returns",

    "Delivery": "delivery",

    "Refunds": "refunds",

    "Sellers": "sellers",

    "Payments": "payments",

    "Customer Support": "customers-support",
}


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="
            text-align:center;
            padding:10px 0 20px 0;
        ">
            <div style="
                font-size:42px;
            ">
                🛍️
            </div>

            <div style="
                font-size:22px;
                font-weight:750;
                color:#f85606;
            ">
                Daraz Assistant
            </div>

            <div style="
                font-size:12px;
                color:#777;
                margin-top:4px;
            ">
                Customer Support Operations
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "### 📚 Knowledge Base"
    )

    selected_department = st.radio(
        "Restrict search to:",
        list(DEPARTMENTS.keys()),
        index=0,
        label_visibility="collapsed",
    )

    department_filter = DEPARTMENTS[
        selected_department
    ]

    st.divider()

    st.markdown(
        "### ⚙️ Retrieval Settings"
    )

    retrieval_k = st.slider(
        "Number of chunks",
        min_value=3,
        max_value=10,
        value=TOP_K,
        step=1,
    )

    st.divider()

    st.markdown(
        "### 📊 System Status"
    )

    st.markdown(
        '<span class="status-online">'
        '● Knowledge Base Loaded'
        '</span>',
        unsafe_allow_html=True,
    )

    st.caption(
        f"{index.ntotal:,} vectors indexed"
    )

    st.caption(
        f"{len(metadata):,} metadata records"
    )

    st.caption(
        "Embeddings are not regenerated."
    )

    st.divider()

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_chunks(
    query,
    department=None,
    k=5
):

    # --------------------------------------------------------
    # Create query embedding
    # --------------------------------------------------------

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    # --------------------------------------------------------
    # Retrieve more candidates when filtering
    # --------------------------------------------------------

    if department:

        search_k = min(
            max(k * 10, 50),
            index.ntotal
        )

    else:

        search_k = min(
            k,
            index.ntotal
        )

    scores, indices = index.search(
        query_embedding,
        search_k
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx < 0:
            continue

        if idx >= len(metadata):
            continue

        item = metadata[idx]

        # ----------------------------------------------------
        # Department filtering
        # ----------------------------------------------------

        if department:

            item_department = (
                item.get(
                    "department",
                    ""
                )
                .lower()
                .strip()
            )

            if item_department != department:

                continue

        result = {
            "score": float(score),
            "id": item.get("id"),
            "department":
                item.get(
                    "department",
                    "unknown"
                ),
            "source_file":
                item.get(
                    "source_file",
                    "unknown"
                ),
            "source_path":
                item.get(
                    "source_path",
                    ""
                ),
            "page_start":
                item.get(
                    "page_start"
                ),
            "page_end":
                item.get(
                    "page_end"
                ),
            "chunk_index":
                item.get(
                    "chunk_index"
                ),
            "text":
                item.get(
                    "text",
                    ""
                ),
        }

        results.append(
            result
        )

        if len(results) >= k:
            break

    return results


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(results):

    context_parts = []

    for i, result in enumerate(
        results,
        start=1
    ):

        source = result[
            "source_file"
        ]

        department = result[
            "department"
        ]

        page_start = result[
            "page_start"
        ]

        page_end = result[
            "page_end"
        ]

        if page_start == page_end:

            pages = f"Page {page_start}"

        else:

            pages = (
                f"Pages "
                f"{page_start}-{page_end}"
            )

        context_parts.append(

            f"""
SOURCE {i}
Department: {department}
File: {source}
{pages}

Content:
{result["text"]}
"""
        )

    return "\n\n".join(
        context_parts
    )


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    query,
    results
):

    if not results:

        return (
            "I couldn't find relevant information "
            "in the selected Daraz knowledge base "
            "section to answer this question."
        )

    context = build_context(
        results
    )

    system_prompt = """
You are the Daraz Customer Support Operations Assistant.

Your job is to answer customer-support and
e-commerce policy questions using ONLY the
provided knowledge-base context.

STRICT RAG RULES:

1. Use only information contained in the
   provided context.

2. Do not invent Daraz policies, prices,
   deadlines, procedures, eligibility rules,
   refund conditions, delivery promises,
   seller requirements, or payment information.

3. If the answer is not clearly available
   in the provided context, say:

   "I couldn't find enough information in
   the Daraz knowledge base to answer that
   accurately."

4. Do not pretend that general knowledge is
   a Daraz policy.

5. Give a clear and practical answer.

6. If the policy contains conditions or
   exceptions, mention them.

7. When useful, use short bullet points.

8. Do not mention the internal retrieval
   process, embeddings, FAISS, chunks, or
   prompts.

9. At the end, include a concise
   "Sources" section using the source
   numbers provided in the context.

10. Never fabricate a source.

11. If multiple sources support the answer,
    cite all relevant source numbers.

12. Keep the answer professional and
    customer-support friendly.
"""

    user_prompt = f"""
Knowledge Base Context:

{context}

Customer Question:

{query}

Answer the customer using only the
knowledge base context above.
"""

    response = groq_client.chat.completions.create(

        model=MODEL_NAME,

        messages=[

            {
                "role": "system",
                "content": system_prompt
            },

            {
                "role": "user",
                "content": user_prompt
            }

        ],

        temperature=0.1,

        max_tokens=1200,

        reasoning_effort="medium",

    )

    return (
        response
        .choices[0]
        .message
        .content
    )


# ============================================================
# DISPLAY SOURCES
# ============================================================

def display_sources(results):

    if not results:
        return

    st.markdown(
        "#### 📚 Retrieved Sources"
    )

    for i, result in enumerate(
        results,
        start=1
    ):

        score = result[
            "score"
        ]

        source = result[
            "source_file"
        ]

        department = result[
            "department"
        ]

        page_start = result[
            "page_start"
        ]

        page_end = result[
            "page_end"
        ]

        if page_start == page_end:

            pages = (
                f"Page {page_start}"
            )

        else:

            pages = (
                f"Pages "
                f"{page_start}-{page_end}"
            )

        st.markdown(
            f"""
            <div class="source-card">
                <b>Source {i}</b><br>
                📄 {source}<br>
                🏷️ Department: {department}<br>
                📖 {pages}<br>
                🔎 Similarity: {score:.3f}
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# WELCOME SCREEN
# ============================================================

if not st.session_state.messages:

    st.markdown(
        """
        <div class="info-card">

        <h3>👋 Welcome to Daraz Customer Support</h3>

        <p>
        Ask questions about Daraz policies and
        customer-support operations using the
        knowledge base.
        </p>

        <b>Try asking:</b>

        <ul>
            <li>What is the return policy?</li>
            <li>How long does delivery take?</li>
            <li>What are the refund conditions?</li>
            <li>What payment methods are supported?</li>
            <li>What are the seller requirements?</li>
        </ul>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        if (
            message["role"]
            == "assistant"
            and message.get(
                "sources"
            )
        ):

            display_sources(
                message["sources"]
            )


# ============================================================
# CHAT INPUT
# ============================================================

user_query = st.chat_input(
    "Ask a question about Daraz policies..."
)


if user_query:

    # --------------------------------------------------------
    # Display user message
    # --------------------------------------------------------

    st.session_state.messages.append({

        "role": "user",

        "content": user_query

    })

    with st.chat_message(
        "user"
    ):

        st.markdown(
            user_query
        )

    # --------------------------------------------------------
    # Retrieve
    # --------------------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "Searching Daraz knowledge base..."
        ):

            results = retrieve_chunks(

                query=user_query,

                department=
                    department_filter,

                k=retrieval_k
            )

        # ----------------------------------------------------
        # Generate
        # ----------------------------------------------------

        with st.spinner(
            "Generating answer..."
        ):

            answer = generate_answer(
                user_query,
                results
            )

        st.markdown(
            answer
        )

        display_sources(
            results
        )

    # --------------------------------------------------------
    # Save assistant message
    # --------------------------------------------------------

    st.session_state.messages.append({

        "role": "assistant",

        "content": answer,

        "sources": results

    })
