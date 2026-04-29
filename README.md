# PDF Knowledge Base Chat API

A RAG (Retrieval-Augmented Generation) application built with FastAPI. Upload PDF documents and chat with their content using Pinecone vector search and Gemini AI.

---

## Tech Stack

| Layer | Tool |
|---|---|
| API Framework | FastAPI |
| PDF Extraction | PyMuPDF (fitz) |
| Vector Database | Pinecone |
| Embeddings | Pinecone Inference API (`multilingual-e5-large`) |
| LLM | Google Gemini (`gemini-2.5-flash`) |

---

## Project Structure

```
rag/
├── main.py             # FastAPI app — /upload and /chat endpoints
├── config.py           # Settings loaded from .env
├── models.py           # Pydantic request/response schemas
├── pdf_processor.py    # PDF text extraction and chunking
├── vector_store.py     # Pinecone embed, upsert, and query
├── llm.py              # Gemini answer generation
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
└── README.md           # This file
```

---

## Prerequisites

- Python 3.10 or higher
- Pinecone account and API key → https://pinecone.io
- Google Gemini API key → https://aistudio.google.com

---

## Setup

### 1. Clone or navigate to the project directory

```bash
cd /path/to/rag
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
```

### 3. Activate the virtual environment

**Linux / macOS:**
```bash
source venv/bin/activate
```

**Windows (Command Prompt):**
```cmd
venv\Scripts\activate.bat
```

**Windows (PowerShell):**
```powershell
venv\Scripts\Activate.ps1
```

> You should see `(venv)` prefix in your terminal after activation.

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in your API keys:

```env
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_INDEX_NAME=rag-index
PINECONE_REGION=us-east-1
PINECONE_EMBED_MODEL=multilingual-e5-large
GEMINI_API_KEY=your-gemini-api-key
GEMINI_MODEL=gemini-2.5-flash
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K=5
```

> `PINECONE_INDEX_NAME`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, and `TOP_K` have defaults and are optional.

---

## Running the Application

### Option 1 — Streamlit UI (recommended for testing)

```bash
streamlit run streamlit_app.py
```

Opens at `http://localhost:8501` in your browser automatically.

**Features:**
- Sidebar PDF uploader — drag and drop or browse for a PDF
- Shows how many chunks were indexed after upload
- Tracks all indexed documents in the session
- Chat interface with message history
- Each answer shows collapsible **Sources** with similarity scores
- Clear chat history button

### Option 2 — FastAPI server (REST API)

```bash
uvicorn main:app --reload --port 8000
```

The server starts at `http://localhost:8000`.

| URL | Description |
|---|---|
| `http://localhost:8000/docs` | Swagger UI — interactive API testing |
| `http://localhost:8000/redoc` | ReDoc — alternative API docs |

---

## API Reference

### POST `/upload`

Upload a PDF file to index its content into Pinecone.

**Request:** `multipart/form-data`

| Field | Type | Description |
|---|---|---|
| `file` | File | PDF file to upload |

**Response:**

```json
{
  "filename": "document.pdf",
  "chunks_stored": 42,
  "message": "PDF indexed successfully."
}
```

**cURL example:**

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@/path/to/your/document.pdf"
```

---

### POST `/chat`

Ask a question based on uploaded PDF content.

**Request:** `application/json`

```json
{
  "question": "What is the main topic of the document?"
}
```

**Response:**

```json
{
  "answer": "The document is about ...",
  "sources": [
    "document.pdf (score: 0.921)",
    "document.pdf (score: 0.887)"
  ]
}
```

**cURL example:**

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic of the document?"}'
```

---

## How It Works

```
POST /upload
  PDF file
    └─ Extract text (PyMuPDF)
         └─ Split into 500-char chunks with 50-char overlap
              └─ Embed chunks (Pinecone Inference — multilingual-e5-large)
                   └─ Store vectors + metadata in Pinecone index

POST /chat
  Question (text)
    └─ Embed question (Pinecone Inference — query mode)
         └─ Search Pinecone for top 5 similar chunks
              └─ Send context + question to Gemini
                   └─ Return answer + source references
```

---

## Deactivating the Virtual Environment

When you are done working:

```bash
deactivate
```

---

## Verify API Keys Before Running

### Check Gemini key is working

```bash
source venv/bin/activate
python3 -c "
from dotenv import load_dotenv; import os; load_dotenv()
from google import genai
client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
print(client.models.generate_content(model='gemini-2.5-flash', contents='Say hi').text)
"
```

If you see `Hi!` — key is valid.

> **Get a Gemini key:** https://aistudio.google.com/app/apikey
> Valid keys always start with `AIza` (capital A, capital I). If your key starts with `Aiza` (lowercase i), it was copied with a typo — get a fresh copy.

### Check Pinecone key is working

```bash
python3 -c "
from dotenv import load_dotenv; import os; load_dotenv()
from pinecone import Pinecone
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
print('Indexes:', [i.name for i in pc.list_indexes().indexes])
"
```

If you see `Indexes: [...]` without error — key is valid.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError` | Ensure venv is activated and `pip install -r requirements.txt` was run |
| `ValidationError` for settings | Check that `.env` file exists and both API keys are filled in |
| `import fitz` fails | PyMuPDF installs as `pymupdf` but imports as `fitz` — this is expected |
| Pinecone index creation slow | First run creates the index; wait ~10–15 seconds, it polls automatically |
| PDF returns 0 chunks | PDF may be image-based (scanned). Only text-based PDFs are supported |
| Gemini 429 error | Free tier rate limit hit — wait a moment and retry |
| Gemini `API key not valid` after fixing `.env` | Restart the Streamlit app — it caches the old key in memory until restarted |
| Gemini key rejected despite correct format | Key may not be activated yet — wait 1–2 minutes after creation and retry |
