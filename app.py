from flask import Flask, render_template, jsonify, request
from src.helper import download_gugging_face_embeddings
from src.prompt import build_system_prompt, build_user_prompt
from langchain_pinecone import PineconeVectorStore
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from googleapiclient.discovery import build
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.memory import ConversationBufferWindowMemory
import os


app = Flask(__name__)

load_dotenv()

GOOGLE_CSE_ID  = os.getenv("GOOGLE_CSE_ID")
GOOGLE_CSE_API_KEY = os.getenv("GOOGLE_CSE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

embeddings = download_gugging_face_embeddings()

index_name = "carebot"

# Embed each chunk and upsert the embeddings into your Pinecone index.                        
docsearch = PineconeVectorStore.from_existing_index(
    index_name= index_name,
    embedding= embeddings
)

retriever = docsearch.as_retriever(search_type = "similarity", search_kwargs = {'k':3})

# Use the API key explicitly
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    # "openai/gpt-oss-20b"
    temperature=0.3,
    groq_api_key=GROQ_API_KEY
)

def google_search(query, api_key, cse_id, num_results=5):
    service = build("customsearch", "v1", developerKey=api_key)
    res     = service.cse().list(q=query, cx=cse_id, num=num_results).execute()
    items   = res.get("items", [])
    return [
        {"title": item.get("title"), "snippet": item.get("snippet"), "link": item.get("link")}
        for item in items
    ]




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

    # Build context with source labels
    content_pieces = []
    for r in context:
        if isinstance(r, dict):
            content_pieces.append(
                f"Source: Web Search\nTitle: {r.get('title','')}\nSnippet: {r.get('snippet','')}\nLink: {r.get('link','')}"
            )
        else:
            content_pieces.append(f"Source: PDF\n{r.page_content}")
    context_str = "\n\n".join(content_pieces)

    # === GET CHAT HISTORY (Last 5 exchanges) ===
    history = memory.load_memory_variables({})["history"]
    history_str = ""
    for msg in history:
        role = "User" if msg.type == "human" else "Assistant"
        history_str += f"{role}: {msg.content}\n"

    # === BUILD PROMPTS ===
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

    # === INVOKE LLM WITH SYSTEM + HISTORY + USER ===
    response = llm.invoke([
        ("system", system_prompt),
        ("user", user_prompt)
    ])

    return response.content



# Initialize memory
memory = ConversationBufferWindowMemory(
    k=5, 
    return_messages=True
)

@app.route("/")
def index():
    return render_template("chat.html")




@app.route("/get", methods=["GET", "POST"])
def chat():
    user_query = request.form["msg"]

    # Generate answer (uses memory)
    ai_response = answer_query(user_query)

    # Save to memory AFTER generating answer
    memory.save_context({"input": user_query}, {"output": ai_response})

    # Optional: debug print
    last_5 = memory.load_memory_variables({})["history"]
    print("Last 5 chat history:\n", last_5)

    return ai_response





# @app.route("/get", methods = ["GET", "POST"])
# def chat():
#     user_query = request.form["msg"]

#     # Generate answer
#     ai_response = answer_query(user_query)

#     # Save to memory
#     memory.save_context({"input": user_query}, {"output": ai_response})

#     # Optional: build last 5 exchanges as context for future LLM calls
#     last_5_history = memory.load_memory_variables({})["history"]
#     print("Last 5 chat history:\n", last_5_history)

#     return ai_response



if __name__ == "__main__":
    app.run(host="0.0.0.0", port="8080", debug=True)