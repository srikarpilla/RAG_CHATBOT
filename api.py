from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_engine import RAGChatbot
import os

app = FastAPI(title="Gemini RAG Chatbot API")

# Initialize Chatbot
chatbot = RAGChatbot()

class QueryRequest(BaseModel):
    query: str

@app.on_event("startup")
async def startup_event():
    """Check if vector DB exists on startup."""
    if os.path.exists("faiss_index"):
        try:
            chatbot.setup_chain()
            print("✅ Chatbot is ready with existing database.")
        except Exception as e:
            print(f"⚠️ Could not load database: {e}")
    else:
        print("⚠️ No Vector Store found. Please use /ingest endpoint.")

@app.post("/ingest")
async def ingest_documents():
    """Endpoint to trigger PDF processing."""
    try:
        success = chatbot.ingest_data()
        if success:
            return {"message": "Documents processed and Vector DB created successfully."}
        else:
            return {"message": "No documents found in 'data' folder to process."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_bot(request: QueryRequest):
    """Endpoint to ask questions."""
    try:
        response = chatbot.ask_question(request.query)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Gemini RAG API is running."}