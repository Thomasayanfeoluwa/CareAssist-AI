from Flask import Flask, render_template, jsonify, request
from src.helper import download_gugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
from src.prompt import *
from langchain_groq import ChatGroq
from googleapiclient.discovery import build
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
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
    model="openai/gpt-oss-20b",
    temperature=0.4,
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

    # Decide if fallback to web search is needed
    if not pdf_results or len(pdf_results) < 2:
        web_results = google_search(user_query, GOOGLE_CSE_API_KEY, GOOGLE_CSE_ID)
        context     = pdf_results + web_results
    else:
        context = pdf_results

    # Build the context string by handling both types
    content_pieces = []
    for r in context:
        if isinstance(r, dict):
            content_pieces.append(r.get("snippet", ""))
        else:
            # r is a Document object
            content_pieces.append(r.page_content)

    context_str = "\n".join(content_pieces)
    
    # Construct prompt
    prompt = (
        f"User asked: {user_query}\n"
        f"Context:\n{context_str}\n"
        "Answer:"
    )

    # Generate answer
    response = llm.invoke([("user", prompt)])
    return response.content



@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods = ["GET", "POST"])
def chat():
    msg = request.form["msg"]
    response = answer_query(msg)
    print("Response:", response)
    return response

