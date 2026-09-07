"""
Main Streamlit application for the HealthMate RAG Chatbot.

This module renders the frontend user interface, manages chat session states,
handles custom CSS styling, and routes queries to the underlying RAG engine.
"""

import sys
from pathlib import Path

# Add project root directory to sys.path to enable absolute package imports
sys.path.append(str(Path(__file__).resolve().parent))

import streamlit as st

from src.config import CHROMA_PERSIST_DIR
from src.vector_store import collection_exists
from src.rag_engine import HealthRAGChatbot

# Configure basic page metadata and layout parameters
st.set_page_config(
    page_title="HealthMate — Asisten Kesehatan Santai",
    page_icon="🩺",
    layout="centered",
)

# Apply custom CSS rules to enhance UI/UX layout, typography, and color palette
st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg, #FAFDFC 0%, #F0F9F7 100%); }
    .healthmate-header {
        display: flex; align-items: center; gap: 12px;
        padding: 18px 22px; border-radius: 16px;
        background: linear-gradient(135deg, #0EA5A5 0%, #14B8A6 100%);
        color: white; margin-bottom: 6px;
    }
    .healthmate-header h1 { margin: 0; font-size: 1.5rem; }
    .healthmate-header p { margin: 0; opacity: 0.9; font-size: 0.9rem; }
    .disclaimer-box {
        background: #FFF7ED; border: 1px solid #FDBA74; color: #7C2D12;
        padding: 10px 14px; border-radius: 10px; font-size: 0.82rem; margin-bottom: 18px;
    }
    .source-chip {
        display: inline-block; background: #E6FFFB; color: #0F766E;
        border-radius: 999px; padding: 2px 10px; font-size: 0.75rem; margin: 2px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Render main header banner
st.markdown(
    """
    <div class="healthmate-header">
        <div style="font-size:2rem;">🩺</div>
        <div>
            <h1>HealthMate</h1>
            <p>Asisten Q&A kesehatan umum berbasis RAG + Gemini API — santai tapi tetap akurat</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Render medical disclaimer box
st.markdown(
    """
    <div class="disclaimer-box">
        ⚠️ <b>Disclaimer:</b> HealthMate bukan pengganti dokter. Informasi bersifat umum
        berdasarkan basis data <b>MedQuAD (NIH)</b>. Untuk gejala serius/darurat, segera
        konsultasi ke tenaga medis profesional.
    </div>
    """,
    unsafe_allow_html=True,
)

# Render side panel for architecture information and status checks
with st.sidebar:
    st.header("ℹ️ Tentang HealthMate")
    st.markdown(
        "Asisten kesehatan berbasis RAG MedQuAD & Gemini API."
    )

    if not collection_exists():
        st.error(
            "Index belum dibuat. Jalankan:\n\n"
            "`python scripts/download_data.py`\n\n"
            "`python scripts/build_index.py`"
        )
        
    else:
        st.success("Knowledge base siap ✅")

    # Action control to reset user conversation state
    if st.button("🗑️ Reset percakapan"):
        st.session_state.messages = []
        st.rerun()

# Initialize message history array in session state if not already present
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": "user"/"model", "text": str, "sources": [...]}

# Instantiate the RAG engine client and catch initialization errors
if "chatbot" not in st.session_state:
    try:
        st.session_state.chatbot = HealthRAGChatbot()
        st.session_state.init_error = None

    except Exception as e:
        st.session_state.chatbot = None
        st.session_state.init_error = str(e)

# Display environment warning if client initialization failed
if st.session_state.init_error:
    st.warning(f"Belum siap: {st.session_state.init_error}")

# Render active chat history stack on page reload
for msg in st.session_state.messages:
    avatar = "🧑" if msg["role"] == "user" else "🩺"

    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["text"])

        if msg.get("sources"):
            with st.expander("📚 Lihat sumber jawaban (MedQuAD/NIH)"):

                for src in msg["sources"]:
                    st.markdown(
                        f"<span class='source-chip'>{src.get('focus') or src.get('qtype') or 'Topik'}</span>",
                        unsafe_allow_html=True,
                    )
                    st.caption(f"**Q:** {src['question']}")
                    st.caption(f"**A:** {src['answer'][:400]}{'...' if len(src['answer']) > 400 else ''}")
                    st.divider()

# Handle new user input turn
user_query = st.chat_input("Tanya apa aja soal kesehatan umum di sini...")

if user_query:
    # Append user turn to local session history
    st.session_state.messages.append({"role": "user", "text": user_query})

    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_query)

    # Process and render assistant model turn
    with st.chat_message("model", avatar="🩺"):
        if not collection_exists():
            reply = "Knowledge base belum di-index. Jalankan script build_index.py dulu ya."
            sources = []
            st.markdown(reply)

        elif st.session_state.chatbot is None:
            reply = f"Belum bisa jalan: {st.session_state.init_error}"
            sources = []
            st.markdown(reply)
            
        else:
            with st.spinner("HealthMate lagi mikir... 🤔"):
                # Extract conversational history omitting current user query
                history_for_engine = [
                    {"role": m["role"], "text": m["text"]}
                    for m in st.session_state.messages[:-1]
                ]

                # Trigger end-to-end RAG workflow execution
                result = st.session_state.chatbot.ask(user_query, history_for_engine)
                reply = result["answer"]
                sources = result["sources"]

            st.markdown(reply)

            # Display retrieved reference documents in expander
            if sources:
                with st.expander("📚 Lihat sumber jawaban (MedQuAD/NIH)"):
                    for src in sources:
                        st.markdown(
                            f"<span class='source-chip'>{src.get('focus') or src.get('qtype') or 'Topik'}</span>",
                            unsafe_allow_html=True,
                        )

                        st.caption(f"**Q:** {src['question']}")
                        st.caption(f"**A:** {src['answer'][:400]}{'...' if len(src['answer']) > 400 else ''}")
                        st.divider()

    # Commit assistant turn and cited context to permanent session state
    st.session_state.messages.append(
        {"role": "model", "text": reply, "sources": sources}
    )
