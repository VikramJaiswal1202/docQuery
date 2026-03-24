from fastapi import FastAPI, UploadFile, File
import os

from app.pdf_loader import load_pdf
from app.vector_store import create_vector_store

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/")
def read_root():
    return {"message": "ChatPDF backend is running"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    documents = load_pdf(file_path)

    chunks = create_vector_store(documents)

    return {
        "message": "PDF processed",
        "pages": len(documents),
        "chunks_created": chunks
    }