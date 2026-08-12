import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from app.services.rag_ingestion import ingest_multimodal_pdf
from app.services.rag_query import query_rag_pipeline

app = FastAPI(
    title="Bank Mandiri Multimodal RAG API",
    description="API for ingesting Bank Mandiri 2025 PDF and querying financial report data.",
    version="1.0.0"
)

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[int]

@app.get("/")
def read_root():
    return {"message": "Bank Mandiri RAG API is running."}

@app.post("/ingest", summary="Upload and Process PDF File")
async def ingest_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    temp_pdf_path = "data/mandiri_report.pdf"
    
    # Save uploaded file
    with open(temp_pdf_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # Run ingestion pipeline
        ingest_multimodal_pdf()
        return {"status": "success", "message": f"Successfully ingested {file.filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@app.post("/query", response_model=QueryResponse, summary="Query the RAG system")
async def query_pdf(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
        
    try:
        result = query_rag_pipeline(request.question)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")