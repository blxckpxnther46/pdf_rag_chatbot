# PDF RAG Knowledge Bot

**Live app:** Local Streamlit App (paste your free Gemini API key in the sidebar)

## Topic
This chatbot is a document retrieval-augmented generation (RAG) assistant that answers queries strictly based on provided PDF, TXT, or Markdown documents.

## Documents
| File | Where it came from (link) |
|---|---|
| sample.md | Built-in starter documentation |

## What I added
- Clean modular design separating backend logic (`rag_core.py` / `rag_backend.py`) and Streamlit UI (`app.py`).
- Automatic pre-population of API key from `.env` if available, along with secure in-browser input.
- Dark theme configuration via `.streamlit/config.toml`.
- Grounded citations linking answers directly back to specific document sources and page numbers.

## Test results
Starter bot verified locally.

## Architecture & Answers
1. **What happens between a question and an answer in my bot?**
   - The user inputs a query in the Streamlit chat UI.
   - The query is embedded using `GoogleGenerativeAIEmbeddings` (`models/gemini-embedding-001`).
   - Relevant chunks are retrieved from the vector store via similarity search (`top_k=4`).
   - Retrieved chunks are assembled into a context prompt with citation markers (`[1]`, `[2]`, etc.).
   - Gemini (`gemini-2.5-flash`) generates the grounded answer with citations or emits the fallback phrase if not found in the documents.

2. **Why did I choose this chunk size and overlap?**
   - `CHUNK_SIZE = 800`: Balanced size preserving semantic context across paragraphs while fitting comfortably within the model's retrieval window.
   - `CHUNK_OVERLAP = 100`: Ensures facts straddling boundary lines are not split and lost during text chunking.
