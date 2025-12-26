# cbh8.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings # Harus sama dengan ingest.py
import gradio as gr

# Load environment
load_dotenv()
# Pastikan di .env Anda, OPENROUTER_API_KEY adalah key yang dari OpenRouter tadi
api_key = os.getenv("OPENROUTER_API_KEY") 

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "example_collection"

# 1. LOAD EMBEDDING (Sama persis dengan ingest.py agar bisa membaca DB)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. LOAD VECTOR DATABASE
vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_PATH,
)

# 3. KONFIGURASI CHAT MODEL (OpenRouter)
# Kita gunakan ChatOpenAI tapi diarahkan ke server OpenRouter
llm = ChatOpenAI(
    openai_api_key=api_key,
    openai_api_base="https://openrouter.ai/api/v1",
    model="google/gemini-2.0-flash-exp:free", # Gunakan model gratis OpenRouter
)

# 4. FUNGSI UNTUK CHAT
def chat_function(message, history):
    # Cari 3 potongan teks paling relevan dari dokumen Anda
    docs = vector_store.similarity_search(message, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Gabungkan pertanyaan user dengan konteks dari dokumen (RAG)
    prompt = f"""Anda adalah asisten cerdas. Gunakan konteks di bawah ini untuk menjawab pertanyaan. 
    Jika tidak ada di konteks, katakan Anda tidak tahu.
    
    KONTEKS:
    {context}
    
    PERTANYAAN:
    {message}
    """
    
    response = llm.invoke(prompt)
    return response.content

# 5. TAMPILAN GRADIO
demo = gr.ChatInterface(
    fn=chat_function, 
    title="Chatbot H8 - Knowledge Base",
    description="Tanya apa saja berdasarkan dokumen yang sudah di-ingest."
)

if __name__ == "__main__":
    print("Sistem siap! Membuka antarmuka Gradio...")
    demo.launch()
