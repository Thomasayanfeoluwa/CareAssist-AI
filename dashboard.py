import streamlit as st
from src.helper import download_gugging_face_embeddings
from src.prompt import build_system_prompt
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from googleapiclient.discovery import build
from langchain_classic.memory import ConversationBufferWindowMemory
from dotenv import load_dotenv
import os

# === Load Environment ===
load_dotenv()

GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
GOOGLE_CSE_API_KEY = os.getenv("GOOGLE_CSE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# === Initialize Components ===
@st.cache_resource
def load_resources():
    embeddings = download_gugging_face_embeddings()
    docsearch = PineconeVectorStore.from_existing_index(
        index_name="carebot", embedding=embeddings
    )
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={'k': 3})

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        groq_api_key=GROQ_API_KEY
    )

    memory = ConversationBufferWindowMemory(k=5, return_messages=True)
    return docsearch, llm, memory

docsearch, llm, memory = load_resources()

# === Google Search Function ===
def google_search(query, api_key, cse_id, num_results=5):
    try:
        service = build("customsearch", "v1", developerKey=api_key)
        res = service.cse().list(q=query, cx=cse_id, num=num_results).execute()
        items = res.get("items", [])
        return [
            {"title": item.get("title"), "snippet": item.get("snippet"), "link": item.get("link")}
            for item in items
        ]
    except Exception as e:
        st.error(f"Web search failed: {e}")
        return []

# === Answer Query with Memory + RAG ===
def answer_query(user_query: str):
    # Retrieve from PDFs
    pdf_results = docsearch.similarity_search(user_query, k=3)

    # Web fallback
    web_results = []
    if not pdf_results or len(pdf_results) < 2:
        web_results = google_search(user_query, GOOGLE_CSE_API_KEY, GOOGLE_CSE_ID)

    context = pdf_results + web_results

    if not context:
        return (
            "I'm sorry, but I do not have enough information from the available sources to answer your question. "
            "Please consult a qualified health professional for guidance."
        )

    # Build context string
    content_pieces = []
    for r in context:
        if isinstance(r, dict):
            content_pieces.append(
                f"**Source: Web Search**\n**Title:** {r.get('title','')}\n**Snippet:** {r.get('snippet','')}\n**Link:** {r.get('link','')}"
            )
        else:
            content_pieces.append(f"**Source: PDF**\n{r.page_content}")
    context_str = "\n\n---\n\n".join(content_pieces)

    # Get chat history
    history = memory.load_memory_variables({})["history"]
    history_str = ""
    for msg in history:
        role = "User" if msg.type == "human" else "Assistant"
        history_str += f"{role}: {msg.content}\n"

    # Build prompt
    system_prompt = build_system_prompt()
    user_prompt = (
        f"### Chat History (Last 5 exchanges):\n{history_str}\n"
        f"### Current User Question:\n{user_query}\n\n"
        f"### Context (PDFs + Web Search):\n{context_str}\n\n"
        "Answer only the current question using the context and history. "
        "Be professional, clear, and cite sources. "
        "End with a reminder to consult a doctor.\n"
        "Answer:"
    )

    # Invoke LLM
    response = llm.invoke([
        ("system", system_prompt),
        ("user", user_prompt)
    ])

    return response.content

# === Streamlit UI ===
st.set_page_config(page_title="CareAssist AI", layout="centered")

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #f0f2f5; }
    .chat-container {
        max-width: 600px;
        margin: auto;
        background: white;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        overflow: hidden;
        display: flex;
        flex-direction: column;
        height: 80vh;
    }
    .header {
        background: #007bff;
        color: white;
        padding: 12px 16px;
        display: flex;
        align-items: center;
        gap: 12px;
        font-weight: 600;
    }
    .user_img {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        border: 2px solid rgba(255,255,255,0.3);
        position: relative;
    }
    .online_icon {
        position: absolute;
        bottom: 0;
        right: 0;
        width: 12px;
        height: 12px;
        background: #28a745;
        border: 2px solid white;
        border-radius: 50%;
        transform: translate(20%, 20%);
    }
    .chat-box {
        flex: 1;
        padding: 16px;
        overflow-y: auto;
        display: flex;
        flex-direction: column;
        gap: 12px;
    }
    .message {
        max-width: 80%;
        padding: 10px 14px;
        border-radius: 18px;
        line-height: 1.5;
        word-wrap: break-word;
    }
    .user { align-self: flex-end; background: #007bff; color: white; border-bottom-right-radius: 4px; }
    .bot { align-self: flex-start; background: #e9ecef; color: #333; border-bottom-left-radius: 4px; }
    .input-area {
        padding: 12px;
        background: #f8f9fa;
        border-top: 1px solid #dee2e6;
    }
    .stTextInput > div > div > input {
        border-radius: 25px;
        padding: 12px 16px;
        border: 1px solid #ced4da;
    }
    .stButton > button {
        border-radius: 25px;
        background: #007bff;
        color: white;
        border: none;
        padding: 0 20px;
        margin-left: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Chat container
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Header
st.markdown("""
<div class="header">
    <div style="position: relative;">
        <img src="https://cdn-icons-png.flaticon.com/512/387/387569.png" class="user_img">
        <span class="online_icon"></span>
    </div>
    <div>CareAssist AI</div>
</div>
""", unsafe_allow_html=True)

# Chat box
chat_box = st.container()
with chat_box:
    st.markdown('<div class="chat-box">', unsafe_allow_html=True)
    for msg in st.session_state.get("messages", []):
        role = "user" if msg["role"] == "user" else "bot"
        st.markdown(f'<div class="message {role}">{msg["content"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Input area
st.markdown('<div class="input-area">', unsafe_allow_html=True)
with st.form(key="chat_form", clear_on_submit=True):
    cols = st.columns([5, 1])
    with cols[0]:
        user_input = st.text_input("Type your message...", key="input", label_visibility="collapsed")
    with cols[1]:
        submit = st.form_submit_button("Send")

if submit and user_input:
    # Add user message
    if "messages" not in st.session_state:
        st.session_state.messages = []
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Generate AI response
    with st.spinner("Thinking..."):
        ai_response = answer_query(user_input)
        st.session_state.messages.append({"role": "bot", "content": ai_response})

        # Save to memory
        memory.save_context({"input": user_input}, {"output": ai_response})

    st.rerun()

st.markdown('</div></div>', unsafe_allow_html=True)

# Initial message
if not st.session_state.get("messages"):
    st.session_state.messages = [
        {"role": "bot", "content": "Hello! Ask me anything about your health. I am your Care Assistant."}
    ]
    st.rerun()