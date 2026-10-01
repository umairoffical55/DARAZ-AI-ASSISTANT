import os
import json
import numpy as np
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
from groq import Groq

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Daraz Customer AI Assistant",
    page_icon="🛍️",
    layout="centered"
)

st.title("🛍️ Daraz Customer Support Operations Assistant")
st.caption("AI-powered support assistant grounded in the Daraz Knowledge Base.")

# ============================================================
# FILE PATHS (Root directory level)
# ============================================================
FAISS_PATH = "index.faiss"
METADATA_PATH = "metadata.json"
CONFIG_PATH = "config.json"

# ============================================================
# LOAD FAISS & EMBEDDING RESOURCES
# ============================================================
@st.cache_resource
def load_rag_resources():
    if not os.path.exists(FAISS_PATH) or not os.path.exists(METADATA_PATH):
        st.error(f"⚠️ Vector database files milti nahi hain. Ensure '{FAISS_PATH}' and '{METADATA_PATH}' are in repository root.")
        return None, None, None, None

    config = {}
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)
    
    model_name = config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")

    # Load SentenceTransformer & FAISS
    model = SentenceTransformer(model_name)
    index = faiss.read_index(FAISS_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    return model, index, chunks, config

with st.spinner("RAG System initialize ho raha hai..."):
    model, index, chunks, config = load_rag_resources()

if model is None or index is None or chunks is None:
    st.stop()

# ============================================================
# GROQ INITIALIZATION
# ============================================================
groq_api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY"))
groq_client = Groq(api_key=groq_api_key) if groq_api_key else None

# ============================================================
# SEARCH FUNCTION
# ============================================================
def search_knowledge_base(query, top_k=3):
    query_vector = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    query_vector = query_vector.astype("float32")

    distances, indices = index.search(query_vector, top_k)

    results = []
    for idx, score in zip(indices[0], distances[0]):
        if idx < len(chunks):
            chunk_data = chunks[idx].copy()
            chunk_data["score"] = float(score)
            results.append(chunk_data)
            
    return results

# ============================================================
# CHAT INTERFACE
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User prompt
if user_query := st.chat_input("Apna sawal yahan type karein..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Searching Daraz Knowledge Base..."):
            retrieved_chunks = search_knowledge_base(user_query, top_k=3)
            context_str = "\n\n".join([f"--- Context {i+1} ---\n{c['text']}" for i, c in enumerate(retrieved_chunks)])

        # If Groq Key is available -> LLM Response
        if groq_client:
            with st.spinner("Generating answer..."):
                prompt = f"""You are an helpful Daraz Customer Support Assistant.
Answer the user query strictly based on the following context. If the answer is not in the context, say "Mujhe is ki jankari knowledge base mein nahi mili."

Context:
{context_str}

User Question: {user_query}
"""
                chat_completion = groq_client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama-3.3-70b-versatile",
                )
                bot_response = chat_completion.choices[0].message.content
        else:
            # Fallback direct retrieved context display
            bot_response = f"**Retrieved Context Matches:**\n\n{context_str}\n\n*(Note: Add `GROQ_API_KEY` in Streamlit Secrets for AI generated summaries.)*"

        st.markdown(bot_response)
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
