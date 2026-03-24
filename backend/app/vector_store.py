from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

vector_db = None


def create_vector_store(documents):

    global vector_db

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    vector_db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory="./chroma_db"
    )

    return len(chunks)


def get_vector_store():
    return vector_db