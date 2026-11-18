import io
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

EMBED_MODEL = "models/gemini-embedding-001"
CHAT_MODEL = "gemini-2.0-flash"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 4
FALLBACK = "I don't have that information in the documents."

# Candidate chat models to try in case a specific model is unavailable for the API key / region
FALLBACK_MODELS = [
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]


def read_file(name, data):
    """Turn one file (its name and its bytes) into Documents. One Document per PDF page."""
    suffix = Path(name).suffix.lower()
    docs = []
    if suffix == ".pdf":
        for number, page in enumerate(PdfReader(io.BytesIO(data)).pages, 1):
            text = page.extract_text() or ""
            if text.strip():
                docs.append(Document(page_content=text, metadata={"source": name, "page": number}))
    elif suffix in (".txt", ".md"):
        docs.append(Document(page_content=data.decode("utf-8", errors="ignore"), metadata={"source": name, "page": 1}))
    return docs


def load_folder(folder="knowledge_base"):
    docs = []
    folder_path = Path(folder)
    if folder_path.exists():
        for path in sorted(folder_path.glob("*")):
            if path.is_file():
                docs += read_file(path.name, path.read_bytes())
    return docs


def load_uploads(files):
    return [d for f in files for d in read_file(f.name, f.getvalue())]


def build_store(docs, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL, google_api_key=api_key)
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
    return InMemoryVectorStore.from_documents(chunks, embeddings), len(chunks)


def make_llm(api_key, model=CHAT_MODEL):
    return ChatGoogleGenerativeAI(model=model, temperature=0.1, google_api_key=api_key)


def ask(question, store, llm):
    docs = store.as_retriever(search_kwargs={"k": TOP_K}).invoke(question)
    context = "\n\n".join(f"[{i}] {d.page_content}" for i, d in enumerate(docs, 1))
    prompt = f"""You answer questions based ONLY on the numbered sources below.
After each fact, cite the source number like [1].
If the sources do not contain the answer, reply exactly: "{FALLBACK}"

Sources:
{context}

Question:
{question}

Answer:"""

    # Extract API key to support fallback models if the primary model returns 404
    api_key_val = getattr(llm.google_api_key, "get_secret_value", lambda: str(llm.google_api_key))()
    primary_model = getattr(llm, "model", CHAT_MODEL)
    models_to_try = [primary_model] + [m for m in FALLBACK_MODELS if m != primary_model]

    last_err = None
    for model_name in models_to_try:
        try:
            curr_llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.1, google_api_key=api_key_val)
            response = curr_llm.invoke(prompt)
            content = response.content
            if isinstance(content, list):
                text_parts = [
                    part["text"] if isinstance(part, dict) and "text" in part else str(part)
                    for part in content
                ]
                content = "".join(text_parts)
            return content, docs
        except Exception as e:
            last_err = e
            continue

    if last_err:
        raise last_err
    return FALLBACK, docs
