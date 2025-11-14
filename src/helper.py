from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings




# Extract the Data from PDF files in the directory
def load_pdf_file(data):
    loader = DirectoryLoader(
        data,
        glob = "*.pdf",
        loader_cls=PyPDFLoader)
        
    documents = loader.load()
    return documents


# Split the Documents into Chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks


# download the Embedding from HugggingFace
def download_gugging_face_embeddings():
    embeddings= HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
    return embeddings