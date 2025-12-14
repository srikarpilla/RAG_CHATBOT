  AI-Powered RAG Document Assistant

A retrieval-augmented generation (RAG) chatbot that allows users to chat with their PDF documents. Built with Google Gemini 2.5, FastAPI, and Streamlit.

 Project Demo
Watch the full working demo of the application here:  
DEPLOYED WEBSITE LINK:https://ragchatbot-ybbddsjzrpquziut8phhai.streamlit.app/

[Watch Demo Video on Loom](https://www.loom.com/share/92d134caff274a94866bd28f1fbbfdbd)



  Features
 Smart Retrieval: Uses local FAISS vector database and HuggingFace embeddings for unlimited, free document processing.
 LLM Intelligence: Powered by Google Gemini 2.5 Flash for high-speed, accurate answers.
 Dual Architecture:
     Backend: Fast & asynchronous API built with `FastAPI`.
     Frontend: Professional UI built with `Streamlit`.
 No Hallucinations: Answers are strictly grounded in the provided PDF content.

  Tech Stack
 Language: Python 3.10+
 AI Models: Google Gemini 2.5 Flash, all-MiniLM-L6-v2 (Embeddings)
 Frameworks: LangChain v0.3, FastAPI, Streamlit
 Database: FAISS (Local Vector Store)

 How to Run

1.  Install Dependencies
    
    pip install -r requirements.txt
    

2.  Set Up API Key
    Create a `.env` file and add your Google API key:
    text
    GOOGLE_API_KEY=your_api_key_here
    

3.  Start the Backend (Terminal 1)
    
    uvicorn api:app --reload
    

4.  Start the Frontend (Terminal 2)
    
    streamlit run ui.py
    

5.  Use the App
     Upload PDFs to the `data/` folder.
     Click "Process / Ingest Documents" in the sidebar.
     Start chatting!


