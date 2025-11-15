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
    return docsearch, llm, memory, retriever

docsearch, llm, memory, retriever = load_resources()

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

# === Streamlit App ===
st.set_page_config(
    page_title="CareAssist AI - Medical Chatbot",
    page_icon="🏥",
    layout="centered"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Header
st.title("🏥 CareAssist AI")
st.markdown("### Your Personal Medical Assistant")
st.markdown("---")

# Display chat messages
chat_container = st.container()
with chat_container:
    if not st.session_state.messages:
        st.info("👋 Hello! I'm your Care Assistant. Ask me anything about health, symptoms, medications, or general medical information.")
    
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant", avatar="🏥"):
                st.write(message["content"])

# Chat input
if prompt := st.chat_input("Type your health question here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Generate assistant response
    with st.chat_message("assistant", avatar="🏥"):
        with st.spinner("Analyzing your question..."):
            try:
                response = answer_query(prompt)
                st.write(response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # Save to memory
                memory.save_context({"input": prompt}, {"output": response})
                
            except Exception as e:
                error_message = "I apologize, but I'm experiencing technical difficulties. Please try again in a moment."
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About CareAssist")
    st.markdown("""
    I'm an AI medical assistant that can help with:
    
    - 💊 Medication information
    - 🩺 Symptom analysis
    - 📚 General health education
    - 🏥 Medical condition information
    - 🔍 Drug interactions
    
    **Disclaimer:** I provide general health information only. 
    Always consult a qualified healthcare professional for 
    medical advice, diagnosis, or treatment.
    """)
    
    st.markdown("---")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        memory.clear()
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 📊 Chat Info")
    st.write(f"Messages in conversation: **{len(st.session_state.messages)}**")