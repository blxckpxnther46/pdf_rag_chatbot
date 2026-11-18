import os
import streamlit as st
from dotenv import load_dotenv

import rag_core

load_dotenv()
env_api_key = os.getenv("GOOGLE_API_KEY", "")
if env_api_key == "your_actual_gemini_api_key":
    env_api_key = ""

st.set_page_config(page_title="My Knowledge Bot", page_icon="📚")
st.title("📚 My Knowledge Bot")
st.caption("Answers come only from the documents. Bring your own Gemini key and, if you like, your own documents.")

# ---- sidebar: the visitor's key and documents ----
api_key = st.sidebar.text_input(
    "Your Gemini API key",
    value=env_api_key,
    type="password",
    help="Get a free key at aistudio.google.com. It stays in this browser session only."
)
uploads = st.sidebar.file_uploader("Upload your own documents (optional)", type=["pdf", "txt", "md"], accept_multiple_files=True)

if st.sidebar.button("🔄 Reset / Clear Session"):
    st.session_state.clear()
    st.rerun()

if not api_key:
    st.info("Paste your Gemini API key in the sidebar to start. Your key is used only to talk to Gemini and is never saved.")
    st.stop()

if uploads:
    docs = rag_core.load_uploads(uploads)
    source_label = f"your {len(uploads)} uploaded file(s)"
    signature = (api_key, tuple((f.name, f.size) for f in uploads))
else:
    docs = rag_core.load_folder()
    source_label = "the built-in documents"
    signature = (api_key, "built-in")

if not docs:
    st.warning("No readable text found. Use text-based PDFs (where you can select the words), .txt or .md files.")
    st.stop()

# ---- build the index once per key + document set, per visitor ----
if st.session_state.get("signature") != signature or "store" not in st.session_state:
    try:
        with st.spinner("Reading and indexing the documents..."):
            store, n_chunks = rag_core.build_store(docs, api_key)
        st.session_state.update(
            signature=signature,
            store=store,
            llm=rag_core.make_llm(api_key),
            n_chunks=n_chunks,
            history=[]
        )
    except Exception as e:
        st.session_state.pop("signature", None)
        st.error(f"Could not index the documents ({type(e).__name__}: {e}). Check that your key is correct and try again.")
        st.stop()
else:
    # Always keep LLM refreshed with latest key and model config
    st.session_state.llm = rag_core.make_llm(api_key)

st.sidebar.write(f"Searching {source_label}: {len(docs)} pages, {st.session_state.n_chunks} chunks")

# ---- chat ----
if "history" not in st.session_state:
    st.session_state.history = []

for role, text, sources in st.session_state.history:
    with st.chat_message(role):
        st.write(text)
        if sources:
            with st.expander("Sources"):
                for s in sources:
                    st.markdown(f"**{s.metadata['source']}, page {s.metadata['page']}**")
                    st.caption(s.page_content[:300])

question = st.chat_input("Ask a question about the documents")
if question:
    st.session_state.history.append(("user", question, None))
    try:
        answer, found = rag_core.ask(question, st.session_state.store, st.session_state.llm)
    except Exception as e:
        answer, found = f"Something went wrong ({type(e).__name__}: {e}). Please try again in a minute.", []
    st.session_state.history.append(("assistant", answer, found))
    st.rerun()
