# Bank Mandiri AI Engineer Technical Test: Multimodal RAG & Layout Extractor

This repository contains the technical assessment submission for the AI Engineer Intern position. It features a complete backend REST API for document querying and a robust Computer Vision script for presentation slide digitization.

## System Architecture

### Part A: Multimodal RAG API (FastAPI)
A retrieval-augmented generation pipeline built to process the Bank Mandiri 2025 Financial Report.
*   **Multimodal Processing:** Uses `PyMuPDF` to extract text and tables, alongside Google Gemini Vision to interpret chart data.
*   **Table Structure Preservation:** Implements large chunk sizes (4000) via LangChain's `RecursiveCharacterTextSplitter` to ensure complex tabular data remains intact for the LLM.
*   **Vector Search:** Local embeddings generated via Hugging Face (`all-MiniLM-L6-v2`) and stored in ChromaDB.
*   **LLM Synthesis:** Powered by `gemini-1.5-flash` with a custom `StrOutputParser` for clean JSON responses containing page-level citations.

### Part B: Layout-Aware Text Extraction
A standalone Python pipeline that transforms static `.jpg` slides into fully editable, pixel-perfect HTML layouts.
*   **Text Detection:** Utilizes `EasyOCR` for highly stable bounding box extraction.
*   **Dynamic Color Matching:** Implements OpenCV K-Means Clustering (`cv2.kmeans`) to dynamically extract exact font and background colors from the original image pixels.
*   **Base64 HTML Injection:** Automatically inpaints the background to remove static text and encodes the clean image directly into the HTML file via Base64, guaranteeing instant, offline browser rendering without local file permission errors.

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repo-link>
   cd bank-mandiri-rag-pipeline
   ```
2. **Create Virtual Environment:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```
3. **Install Dependencies:**
   ```
   pip install -r requirements.txt
   ```
4. **Environment Variables:**
   Create a .env file in the root directory and add your AI Model API key:
   ```
   AI_MODEL_API_KEY="your_api_key_here"
   ```
# Running the Projects
## Part A Multimodal RAG API (FastAPI)
Start the FastAPI Server
  ```
  uvicorn app.main:app --reload
  ```
Navigate to http://127.0.0.1:8000/docs to access the interactive Swagger UI.
- POST /ingest: Upload the mandiri_report.pdf to populate the ChromaDB vector store.
- POST /query: Ask questions based on the financial report (e.g., querying credit growth percentages).

## Part B Layout-Aware Text Extraction
Ensure your target images are in the slides_presentation/ directory, then run:
  ```
  python app/services/layout_extractor.py
  ```
The script will generate standalone .html files in the same directory. Open them in any modern web browser to view and edit the text.
