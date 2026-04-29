from fastapi import FastAPI, File, HTTPException, UploadFile

from config import settings
from llm import generate_answer
from models import ChatRequest, ChatResponse, UploadResponse
from pdf_processor import chunk_text, extract_text
from vector_store import query_similar, upsert_chunks

app = FastAPI(title="PDF Knowledge Base Chat")


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    file_bytes = await file.read()
    text = extract_text(file_bytes)

    if not text.strip():
        raise HTTPException(status_code=422, detail="PDF has no extractable text (may be a scanned image).")

    chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
    upsert_chunks(chunks, source=file.filename)

    return UploadResponse(
        filename=file.filename,
        chunks_stored=len(chunks),
        message="PDF indexed successfully.",
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    results = query_similar(request.question)

    if not results:
        raise HTTPException(status_code=404, detail="No relevant content found. Upload a PDF first.")

    context_chunks = [r["text"] for r in results]
    sources = [f"{r['source']} (score: {r['score']:.3f})" for r in results]
    answer = generate_answer(request.question, context_chunks)

    return ChatResponse(answer=answer, sources=sources)
