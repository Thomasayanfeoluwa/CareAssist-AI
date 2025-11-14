import os
from dotenv import load_dotenv
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from src.helper import load_pdf_file, text_split, download_gugging_face_embeddings                         


load_dotenv()

# Read API key from environment variable
PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if PINECONE_API_KEY is None:
    raise ValueError("PINECONE_API_KEY environment variable is not set.")

extracted_data=load_pdf_file(data='Data/')
text_chunks = text_split(extracted_data)
embeddings = download_gugging_face_embeddings()

# Initialize client
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "carebot"
vector_dimension = 384 
metric_type = "cosine"

# Check whether the index exists
if not pc.has_index(name=index_name):
    pc.create_index(
        name=index_name,
        dimension=vector_dimension,
        metric=metric_type,
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# To get the index object later:
index = pc.Index(index_name)



# Embed each chunk and upsert the embeddings into your Pinecone index.                        
docsearch = PineconeVectorStore.from_documents(
    documents = text_chunks,
    index_name = index_name,
    embedding = embeddings
)