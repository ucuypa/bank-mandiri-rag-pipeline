import os
import pymupdf  # PyMuPDF
import base64
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage


load_dotenv()
PDF_PATH = "data/mandiri_report.pdf"
CHROMA_DB_DIR = "data/chroma_db"

def summarize_image_with_gemini(image_bytes):
    """Sends an image to Gemini and asks it to describe the chart/data."""
    # Initialize Gemini Vision
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    
    # Encode image to base64
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")
    
    message = HumanMessage(
        content=[
            {"type": "text", "text": "Describe the data, charts, or text in this image in detail. If it is a pie chart, list the percentages and labels."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_image}"}}
        ]
    )
    
    try:
        response = llm.invoke([message])
        return response.content
    except Exception as e:
        print(f"Error summarizing image: {e}")
        return ""

def ingest_multimodal_pdf():
    print(f"Loading PDF from {PDF_PATH} using PyMuPDF...")
    
    doc = pymupdf.open(PDF_PATH)
    all_documents = []
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        page_text = page.get_text("text")
        image_summaries = []
        image_list = page.get_images(full=True)
        
        if image_list:
            print(f"Found {len(image_list)} image(s) on page {page_num + 1}. Summarizing with Gemini...")
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                summary = summarize_image_with_gemini(image_bytes)
                if summary:
                    image_summaries.append(f"[Image Data on Page {page_num + 1}]: {summary}")
        
        # Combine Text and Image Summaries
        combined_content = page_text + "\n\n" + "\n\n".join(image_summaries)
        
        # Create a LangChain Document object
        metadata = {"page": page_num + 1, "source": PDF_PATH}
        all_documents.append(Document(page_content=combined_content, metadata=metadata))

    print(f"Successfully processed {len(all_documents)} pages with text and images.")
    print("Chunking the combined data")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=500,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(all_documents)
    print(f"Split document into {len(chunks)} chunks.")

    print("Loading Hugging Face embeddings")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR
    )
    
    print(f"Success, Multimodal vector database created and saved at {CHROMA_DB_DIR}")

if __name__ == "__main__":
    ingest_multimodal_pdf()