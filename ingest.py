import os
from uuid import uuid4
from dotenv import load_dotenv

from langchain_community.document_loaders import (
    PyPDFDirectoryLoader, 
    DirectoryLoader, 
    TextLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
# Gunakan HuggingFace karena gratis dan tidak butuh API Key
from langchain_huggingface import HuggingFaceEmbeddings 

load_dotenv()

DATA_PATH = "data"
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "example_collection"

# --- Embedding Model (GRATIS & LOKAL) ---
# Model 'all-MiniLM-L6-v2' sangat ringan dan cepat untuk laptop
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# --- Vector Store ---
vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_PATH,
)

# ===== LOAD DOCUMENTS =====
documents = []
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

pdf_loader = PyPDFDirectoryLoader(DATA_PATH)
documents.extend(pdf_loader.load())

md_loader = DirectoryLoader(
    DATA_PATH,
    glob="**/*.md",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"}
)
documents.extend(md_loader.load())

if len(documents) == 0:
    print(f"❌ Folder '{DATA_PATH}' kosong. Isi dengan file PDF atau MD!")
else:
    print(f"Loaded {len(documents)} documents")

    # ===== SPLIT DOCUMENTS =====
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    # ===== STORE IN VECTOR DB =====
    ids = [str(uuid4()) for _ in range(len(chunks))]
    vector_store.add_documents(documents=chunks, ids=ids)
    print("✅ Ingestion complete")
