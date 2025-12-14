import os
import time
from pathlib import Path
from typing import List, Dict, Optional
import hashlib
import json

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()


class RateLimiter:
    
    
    def __init__(self, max_calls_per_minute: int = 10):
        self.max_calls = max_calls_per_minute
        self.calls = []
        self.min_delay = 2
        self.last_call_time = 0
    
    def wait_if_needed(self):
        """Enforce rate limiting."""
        current_time = time.time()
        self.calls = [t for t in self.calls if current_time - t < 60]
        
        if len(self.calls) >= self.max_calls:
            sleep_time = 60 - (current_time - self.calls[0]) + 1
            if sleep_time > 0:
                print(f"⏳ Rate limit approaching. Waiting {sleep_time:.1f}s...")
                time.sleep(sleep_time)
                self.calls = []
        
        time_since_last = current_time - self.last_call_time
        if time_since_last < self.min_delay:
            time.sleep(self.min_delay - time_since_last)
        
        self.calls.append(time.time())
        self.last_call_time = time.time()


class ResponseCache:
    #Cache responses to avoid redundant API calls.
    
    def __init__(self, cache_file: str = "response_cache.json"):
        self.cache_file = cache_file
        self.cache = self._load_cache()
    
    def _load_cache(self) -> Dict:
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_cache(self):
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Warning: Could not save cache: {e}")
    
    def _hash_query(self, query: str, context: str) -> str:
        combined = f"{query}_{context}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def get(self, query: str, context: str) -> Optional[str]:
        key = self._hash_query(query, context)
        return self.cache.get(key)
    
    def set(self, query: str, context: str, response: str):
        key = self._hash_query(query, context)
        self.cache[key] = response
        self._save_cache()


class RAGChatbot:
  
    
    def __init__(self):
        self.data_path = "data"
        self.db_path = "faiss_index"
        self.embeddings = None
        self.vectorstore = None
        self.qa_chain = None
        self.rate_limiter = RateLimiter(max_calls_per_minute=8)
        self.cache = ResponseCache()
        self.llm = None
        
        # Delay embeddings initialization
        print("✅ RAG Chatbot initialized (embeddings will load on first use)")
    
    def _setup_embeddings(self):
        """Initialize embeddings - using FREE local HuggingFace (no API needed)."""
        if self.embeddings is not None:
            return
        
        print("🚀 Loading local embeddings model (100% free, no API calls)...")
        
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            
         
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            print("✅ Local embeddings ready! (No API quota used)")
        except Exception as e:
            print(f"❌ Failed to initialize embeddings: {e}")
            raise
    
    def ingest_data(self) -> bool:
        """Process PDFs and create vector database."""
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
            print(f"📁 Created '{self.data_path}' folder. Please add PDF files.")
            return False
        
        pdf_files = list(Path(self.data_path).glob("*.pdf"))
        if not pdf_files:
            print("⚠️ No PDF files found in 'data' folder.")
            return False
        
        print(f"📚 Found {len(pdf_files)} PDF(s). Starting processing...")
        
        try:
            
            self._setup_embeddings()
            
            # Load documents with progress
            print("📖 Loading PDFs...")
            loader = DirectoryLoader(
                self.data_path,
                glob="*.pdf",
                loader_cls=PyPDFLoader,
                show_progress=False,
                use_multithreading=True
            )
            documents = loader.load()
            print(f"✅ Loaded {len(documents)} pages from {len(pdf_files)} PDF(s)")
            
            # Split documents
            print("✂️ Splitting into chunks...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            texts = text_splitter.split_documents(documents)
            print(f"✅ Created {len(texts)} text chunks")
            
            # Create vector store with rate limiting
            print("🔄 Building vector database (this may take 1-2 minutes)...")
            
            # Process in batches to avoid rate limits
            batch_size = 100
            if len(texts) > batch_size:
                # Create initial batch
                self.vectorstore = FAISS.from_documents(
                    texts[:batch_size], 
                    self.embeddings
                )
                print(f"   Processed {batch_size}/{len(texts)} chunks...")
                
                # Add remaining in batches
                for i in range(batch_size, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    self.vectorstore.add_documents(batch)
                    print(f"   Processed {min(i + batch_size, len(texts))}/{len(texts)} chunks...")
                    time.sleep(1)  # Small delay between batches
            else:
                self.vectorstore = FAISS.from_documents(texts, self.embeddings)
            
            # Save to disk
            print("💾 Saving vector database...")
            self.vectorstore.save_local(self.db_path)
            print(f"✅ Database saved to '{self.db_path}'")
            
            # Setup QA chain
            self.setup_chain()
            return True
            
        except Exception as e:
            print(f"❌ Error during ingestion: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def setup_chain(self):
        """Setup the QA chain with the vector store."""
        try:
            # Load vector store if not already loaded
            if self.vectorstore is None:
                if not os.path.exists(self.db_path):
                    raise FileNotFoundError("Vector database not found. Run /ingest first.")
                
                print("📦 Loading vector database...")
                self._setup_embeddings()
                
                self.vectorstore = FAISS.load_local(
                    self.db_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print("✅ Database loaded")
            
            # Initialize LLM if not already done
            if self.llm is None:
                api_key = os.getenv("GOOGLE_API_KEY")
                if not api_key:
                    raise ValueError("GOOGLE_API_KEY not found.")
                
                print("🤖 Initializing AI model...")
                
                # Configure genai to list available models
                genai.configure(api_key=api_key)
               
                models_to_try = [
                    "models/gemini-2.5-flash",      
                    "models/gemini-2.0-flash",     
                    "models/gemini-flash-latest",   
                    "models/gemini-pro-latest",     
                ]
                
                last_error = None
                for model_name in models_to_try:
                    try:
                        print(f"   Trying: {model_name}...")
                        self.llm = ChatGoogleGenerativeAI(
                            model=model_name,
                            google_api_key=api_key,
                            temperature=0.3,
                            max_output_tokens=1024
                        )
                       
                        test_result = self.llm.invoke("Hi")
                        print(f"✅ AI model ready: {model_name}")
                        break
                    except Exception as e:
                        last_error = e
                        error_str = str(e).lower()
                        if "not found" in error_str or "404" in error_str or "not supported" in error_str:
                            print(f"   ❌ {model_name} not available")
                            continue 
                        else:
                           
                            print(f"   ⚠️ Error with {model_name}: {str(e)[:100]}")
                            raise
                else:
                   
                    print("\n❌ Could not initialize any model.")
                    print("\n🔍 Checking available models for your API key...\n")
                    try:
                        available_models = genai.list_models()
                        print("Available models:")
                        for m in available_models:
                            if 'generateContent' in m.supported_generation_methods:
                                print(f"  - {m.name}")
                    except:
                        pass
                    raise Exception(f"Could not initialize AI model. Last error: {last_error}")
            
            # Custom prompt
            template = """You are a helpful AI assistant that answers questions based strictly on the provided context.

Context from documents:
{context}

Question: {question}

Instructions:
- Answer ONLY using information from the context above
- If the answer is not in the context, say "I don't have enough information in the documents to answer that question."
- Be concise and accurate
- Cite specific details when possible

Answer:"""
            
            PROMPT = PromptTemplate(
                template=template,
                input_variables=["context", "question"]
            )
            
            # Create QA chain
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 3}
                ),
                return_source_documents=True,
                chain_type_kwargs={"prompt": PROMPT}
            )
            
            print("✅ System ready to answer questions!")
            
        except Exception as e:
            print(f"❌ Error setting up chain: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def ask_question(self, query: str, max_retries: int = 3) -> Dict:
        """Ask a question with retry logic and caching."""
        if not self.qa_chain:
            try:
                self.setup_chain()
            except:
                return {
                    "answer": "⚠️ System not initialized. Please ingest documents first using /ingest endpoint.",
                    "sources": []
                }
        
        # Get relevant context for cache key
        try:
            docs = self.vectorstore.similarity_search(query, k=2)
            context_preview = " ".join([doc.page_content[:100] for doc in docs])
        except:
            context_preview = ""
        
        # Check cache first
        cached_response = self.cache.get(query, context_preview)
        if cached_response:
            print("💾 Returning cached response")
            try:
                return json.loads(cached_response)
            except:
                pass  # If cache is corrupted, continue to generate new response
        
        # Attempt with retries
        for attempt in range(max_retries):
            try:
                # Apply rate limiting
                self.rate_limiter.wait_if_needed()
                
                if attempt > 0:
                    print(f"🔄 Retry attempt {attempt + 1}/{max_retries}...")
                
                # Execute query
                result = self.qa_chain.invoke({"query": query})
                
                # Extract answer and sources
                answer = result.get("result", "No answer generated.")
                source_docs = result.get("source_documents", [])
                
                # Get unique source files
                sources = list(set([
                    doc.metadata.get("source", "Unknown")
                    for doc in source_docs
                ]))
                
                response = {
                    "answer": answer,
                    "sources": sources
                }
                
                # Cache the response
                self.cache.set(query, context_preview, json.dumps(response))
                
                return response
                
            except Exception as e:
                error_msg = str(e)
                
                # Handle rate limit errors
                if "429" in error_msg or "quota" in error_msg.lower() or "rate" in error_msg.lower():
                    if attempt < max_retries - 1:
                        wait_time = 60
                        
                        # Try to extract wait time from error
                        try:
                            import re
                            match = re.search(r'retry in (\d+\.?\d*)(ms|s)', error_msg)
                            if match:
                                wait_val = float(match.group(1))
                                unit = match.group(2)
                                wait_time = (wait_val / 1000 if unit == 'ms' else wait_val) + 5
                        except:
                            pass
                        
                        print(f"⏳ Rate limit hit. Waiting {wait_time:.0f}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        return {
                            "answer": (
                                "⚠️ **Google Gemini API Quota Exhausted**\n\n"
                                "Your API key has hit its free tier limits.\n\n"
                                "**IMMEDIATE SOLUTIONS:**\n\n"
                                "1. **Get a NEW API key** (takes 30 seconds):\n"
                                "   → Visit: https://makersuite.google.com/app/apikey\n"
                                "   → Click 'Create API Key'\n"
                                "   → Copy the new key\n"
                                "   → Update your `.env` file\n"
                                "   → Restart the server\n\n"
                                "2. **Wait 24 hours** for your quota to reset\n\n"
                                "3. **Upgrade to paid tier** for unlimited usage:\n"
                                "   → Visit: https://ai.google.dev/pricing\n\n"
                                "💡 **Note**: The embeddings now run locally (free & unlimited), "
                                "only chat responses use the API."
                            ),
                            "sources": []
                        }
                
                # Handle other errors
                if attempt < max_retries - 1:
                    print(f"⚠️ Error: {error_msg[:100]}... Retrying...")
                    time.sleep(2 ** attempt)
                    continue
                else:
                    return {
                        "answer": f"❌ Error: {error_msg}",
                        "sources": []
                    }
        
        return {
            "answer": "❌ Failed after multiple attempts. Please try again later.",
            "sources": []
        }


# For testing
if __name__ == "__main__":
    print("\n" + "="*60)
    print("RAG CHATBOT - QUICK TEST")
    print("="*60 + "\n")
    
    chatbot = RAGChatbot()
    
    if os.path.exists("faiss_index"):
        print("✅ Found existing database\n")
        chatbot.setup_chain()
    else:

        print("⚠️ No database found. Run ingestion first.\n")
