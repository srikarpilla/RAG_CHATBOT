import streamlit as st
import requests
import os
import time
from dotenv import load_dotenv


# 1. CONFIGURATION & STYLING

load_dotenv()
API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Enterprise RAG Assistant",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

#  CSS 
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .stButton button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E1E1E;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# 2. LOCAL LOGIC (GREETINGS & EXIT
def handle_local_intent(user_input):
    """
    Intercepts user input to handle greetings and exits locally
    for a snappier, more natural experience.
    """
    u_in = user_input.lower().strip()
    
    # Greetings
    greetings = ["hi", "hello", "hey", "greetings", "good morning", "good evening"]
    if any(u_in.startswith(g) for g in greetings) and len(u_in) < 20:
        return (
            "Hello! 👋\n\n"
            "I am your **AI Document Assistant**, powered by Google Gemini 2.0. "
            "I can read your uploaded PDF documents and answer questions based strictly on their content.\n\n"
            "**How can I help you today?**"
        )
    
    # Exits / Gratitude
    exits = ["bye", "exit", "quit", "goodbye", "see ya"]
    gratitude = ["thank you", "thanks", "thx"]
    
    if any(u_in == e for e in exits):
        return "Thank you for using the system. Have a productive day! 👋"
    
    if any(u_in.startswith(t) for t in gratitude):
        return "You're very welcome! If you have more questions about your documents, feel free to ask."

    return None


# 3. SIDEBAR CONTROLS

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=50)
    st.markdown("### Control Panel")
    
    # API Key Handling
    if not os.getenv("GOOGLE_API_KEY"):
        st.warning("⚠️ API Key Missing")
        api_key = st.text_input("Enter Google API Key", type="password")
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
            st.success("Key Set!")
    else:
        st.success("✅ API Key Active")

    st.markdown("---")
    
    # Ingestion Controls
    st.markdown("**Document Management**")
    st.info("Ensure PDF files are in the `data/` folder.")
    
    if st.button("🔄 Process / Ingest Documents", type="primary"):
        with st.spinner("Analyzing documents..."):
            try:
                start_time = time.time()
                res = requests.post(f"{API_URL}/ingest")
                duration = round(time.time() - start_time, 2)
                
                if res.status_code == 200:
                    st.success(f"Success! Processed in {duration}s.")
                    st.toast("Knowledge Base Updated", icon="✅")
                else:
                    st.error(f"Error: {res.text}")
            except Exception as e:
                st.error("Backend Connection Failed")
                st.caption(f"Details: {e}")

    st.markdown("---")
    
    # Clear Chat
    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.caption("v1.0.0 | Enterprise Edition")

# 4. MAIN CHAT INTERFACE

st.markdown('<p class="main-header">📄 Enterprise RAG Assistant</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Chat with your internal documents securely and efficiently.</p>', unsafe_allow_html=True)

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant", 
        "content": "Hello! I'm ready to analyze your documents. What would you like to know?"
    })

# Display Chat History
for msg in st.session_state.messages:
    avatar = "🧑‍💻" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# User Input Handler
if prompt := st.chat_input("Type your question here..."):
    
    # 1. Display User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    # 2. Generate Response
    with st.chat_message("assistant", avatar="🤖"):
        # Check for local intent (Hi/Bye) first
        local_response = handle_local_intent(prompt)
        
        if local_response:
            
            st.markdown(local_response)
            st.session_state.messages.append({"role": "assistant", "content": local_response})
        else:
          
            with st.spinner("Analyzing knowledge base..."):
                try:
                    payload = {"query": prompt}
                    res = requests.post(f"{API_URL}/ask", json=payload)
                    
                    if res.status_code == 200:
                        data = res.json()
                        ans = data.get("answer", "I couldn't generate an answer.")
                        src = data.get("sources", [])
                        
                       
                        full_resp = ans
                        if src:
                       
                            clean_sources = [os.path.basename(s) for s in src]
                            full_resp += f"\n\n---\n**📚 References:**\n" + "\n".join([f"- `{s}`" for s in clean_sources])
                        
                        st.markdown(full_resp)
                        st.session_state.messages.append({"role": "assistant", "content": full_resp})
                    
                    elif res.status_code == 429:
                        error_msg = "⚠️ High traffic volume (Rate Limit). Please wait 30 seconds and try again."
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    else:
                        st.error(f"Server Error {res.status_code}")
                
                except Exception as e:
                    st.error("🔌 Backend is unreachable.")

                    st.info("Please ensure `uvicorn api:app --reload` is running.")
