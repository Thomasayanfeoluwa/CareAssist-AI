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
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        groq_api_key=GROQ_API_KEY
    )
    memory = ConversationBufferWindowMemory(k=5, return_messages=True)
    return docsearch, llm, memory

docsearch, llm, memory = load_resources()

# === Google Search ===
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
        st.error(f"Web search error: {e}")
        return []

# === Answer Query ===
def answer_query(user_query: str):
    pdf_results = docsearch.similarity_search(user_query, k=3)
    web_results = []
    if not pdf_results or len(pdf_results) < 2:
        web_results = google_search(user_query, GOOGLE_CSE_API_KEY, GOOGLE_CSE_ID)

    context = pdf_results + web_results
    if not context:
        return "I'm sorry, but I do not have enough information from the available sources to answer your question. Please consult a qualified health professional for guidance."

    content_pieces = []
    for r in context:
        if isinstance(r, dict):
            content_pieces.append(
                f"**Source: Web Search**\n**Title:** {r.get('title','')}\n**Snippet:** {r.get('snippet','')}\n**Link:** {r.get('link','')}"
            )
        else:
            content_pieces.append(f"**Source: PDF**\n{r.page_content}")
    context_str = "\n\n---\n\n".join(content_pieces)

    history = memory.load_memory_variables({})["history"]
    history_str = ""
    for msg in history:
        role = "User" if msg.type == "human" else "Assistant"
        history_str += f"{role}: {msg.content}\n"

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

    response = llm.invoke([
        ("system", system_prompt),
        ("user", user_prompt)
    ])
    return response.content

# === Streamlit Page Config ===
st.set_page_config(page_title="CareAssist AI", layout="centered", initial_sidebar_state="collapsed")

# === Inject Exact HTML/CSS/JS ===
st.markdown("""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>CareAssist AI</title>

  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>

  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      background: #f0f2f5;
      height: 100vh;
      display: flex;
      justify-content: center;
      align-items: center;
    }
    .chat-container {
      width: 100%;
      max-width: 600px;
      background: white;
      border-radius: 12px;
      box-shadow: 0 4px 20px rgba(0,0,0,0.1);
      overflow: hidden;
      display: flex;
      flex-direction: column;
      height: 80vh;
    }
    .chat-header {
      background: #007bff;
      color: white;
      padding: 12px 16px;
      display: flex;
      align-items: center;
      font-size: 1.2rem;
      font-weight: 600;
      gap: 12px;
    }
    .chat-header img {
      width: 40px;
      height: 40px;
      border-radius: 50%;
      object-fit: cover;
      border: 2px solid rgba(255,255,255,0.3);
    }
    .chat-header .title {
      flex: 1;
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
      line-height: 1.4;
      word-wrap: break-word;
    }
    .user {
      align-self: flex-end;
      background: #007bff;
      color: white;
      border-bottom-right-radius: 4px;
    }
    .bot {
      align-self: flex-start;
      background: #e9ecef;
      color: #333;
      border-bottom-left-radius: 4px;
    }
    .input-area {
      display: flex;
      padding: 12px;
      background: #f8f9fa;
      border-top: 1px solid #dee2e6;
    }
    #msg {
      flex: 1;
      padding: 12px 16px;
      border: 1px solid #ced4da;
      border-radius: 25px;
      font-size: 1rem;
      outline: none;
      max-width: none;
      width: 100%;
    }
    #msg:focus {
      border-color: #007bff;
    }
    button {
      margin-left: 8px;
      padding: 0 20px;
      background: #007bff;
      color: white;
      border: none;
      border-radius: 25px;
      font-size: 1rem;
      cursor: pointer;
      transition: 0.2s;
    }
    button:hover {
      background: #0056b3;
    }
    .typing {
      font-style: italic;
      color: #666;
    }
    .message table {
      width: 100%;
      border-collapse: collapse;
      margin: 10px 0;
      font-size: 0.92rem;
    }
    .message th, .message td {
      border: 1px solid #ccc;
      padding: 7px 9px;
      text-align: left;
    }
    .message th {
      background-color: #f5f7fa;
      font-weight: 600;
    }
    .message ul, .message ol {
      margin: 8px 0;
      padding-left: 20px;
    }
    .message li {
      margin: 4px 0;
    }
  </style>
</head>
<body>

  <div class="chat-container">
    <div class="chat-header">
      <img src="static/nurse_avatar.png" alt="Nurse">
      <div class="title">CareAssist AI</div>
    </div>

    <div class="chat-box" id="chatBox">
      <div class="message bot">Hello! Ask me anything about your Health. I am your Care Assistant.</div>
    </div>

    <form class="input-area" id="chatForm">
      <input type="text" id="msg" name="msg" placeholder="Type your message..." autocomplete="off" required />
      <button type="submit">Send</button>
    </form>
  </div>

  <script>
    const chatBox = document.getElementById("chatBox");
    const chatForm = document.getElementById("chatForm");
    const msgInput = document.getElementById("msg");

    const scrollToBottom = () => {
      chatBox.scrollTop = chatBox.scrollHeight;
    };

    const addMessage = (text, type) => {
      const msgDiv = document.createElement("div");
      msgDiv.classList.add("message", type);
      if (type === "bot") {
        msgDiv.innerHTML = marked.parse(text);
      } else {
        msgDiv.textContent = text;
      }
      chatBox.appendChild(msgDiv);
      scrollToBottom();
    };

    chatForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const userText = msgInput.value.trim();
      if (!userText) return;

      addMessage(userText, "user");
      msgInput.value = "";

      const typing = document.createElement("div");
      typing.classList.add("message", "bot", "typing");
      typing.textContent = "Thinking...";
      chatBox.appendChild(typing);
      scrollToBottom();

      try {
        const formData = new FormData();
        formData.append("msg", userText);

        const response = await fetch("/get", { method: "POST", body: formData });
        const botReply = await response.text();

        typing.remove();
        addMessage(botReply, "bot");
      } catch (err) {
        typing.remove();
        addMessage("Error: Could not connect.", "bot");
      }
    });

    scrollToBottom();
  </script>

</body>
</html>
""", unsafe_allow_html=True)

# === Streamlit Backend Logic (Hidden from UI) ===
if "messages" not in st.session_state:
    st.session_state.messages = []

# Handle form submit via Streamlit (simulate /get POST)
user_input = st.text_input("", key="streamlit_input", label_visibility="collapsed")
if st.button("Send", key="send_btn"):
    if user_input.strip():
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Generate response
        with st.spinner(""):
            ai_response = answer_query(user_input)
            st.session_state.messages.append({"role": "bot", "content": ai_response})

            # Save to memory
            memory.save_context({"input": user_input}, {"output": ai_response})

        st.rerun()

# === Render Messages Dynamically (Streamlit) ===
for msg in st.session_state.messages:
    role = "user" if msg["role"] == "user" else "bot"
    if role == "bot":
        st.markdown(f'<div class="message bot">{marked.parse(msg["content"])}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="message user">{msg["content"]}</div>', unsafe_allow_html=True)

# Initial message
if not st.session_state.messages:
    st.session_state.messages = [
        {"role": "bot", "content": "Hello! Ask me anything about your Health. I am your Care Assistant."}
    ]
    st.rerun()