import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

CHROMA_DB_DIR = "data/chroma_db"

def query_rag_pipeline(question: str):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    retrieved_docs = retriever.invoke(question)
    
    if not retrieved_docs:
        return {
            "answer": "Tidak ditemukan konteks yang relevan dalam dokumen.",
            "sources": []
        }
    
    # Extract text content and metadata
    context_text = ""
    sources = []
    
    for doc in retrieved_docs:
        page_num = doc.metadata.get("page", "Unknown")
        context_text += f"\n--- Context from Page {page_num} ---\n{doc.page_content}\n"
        if page_num not in sources:
            sources.append(page_num)
            
    # Define Gemini Prompt
    prompt_template = ChatPromptTemplate.from_template(
        """
        Anda adalah asisten AI yang membantu menjawab pertanyaan berdasarkan Laporan Bank Mandiri 2025.
        Jawablah pertanyaan berikut hanya berdasarkan konteks yang diberikan. 
        Jika jawaban tidak terdapat dalam konteks, katakan bahwa informasi tidak tersedia secara jelas.

        Konteks:
        {context}

        Pertanyaan: {question}

        Jawaban Singkat & Akurat:
        """
    )
    
    # Synthesize Answer using Gemini
    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.2)
    chain = prompt_template | llm | StrOutputParser()
    
    response_text = chain.invoke({"context": context_text, "question": question})
    
    return {
        "question": question,
        "answer": response_text,
        "sources": sources
    }