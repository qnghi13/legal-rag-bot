import os
import pickle
import sys
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever

# Đảm bảo import được ingest.py cùng thư mục
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ingest import load_and_chunk_folder

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def create_vector_db(
    folder_path: str,
    *,
    # Chunking
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    # Retrievers
    bm25_k: int = 10,
    # Embedding
    embedding_model: str = "keepitreal/vietnamese-sbert",
    # Paths
    chroma_path: str | None = None,
    bm25_path: str | None = None,
):
    """Tạo Chroma Vector DB và BM25 retriever từ thư mục chứa PDF/txt/md.

    Parameters
    ----------
    folder_path : str
        Đường dẫn thư mục chứa tài liệu đầu vào.
    chunk_size : int
        Kích thước chunk (số ký tự) khi split text (mặc định 1000).
    chunk_overlap : int
        Số ký tự overlap giữa các chunk (mặc định 200).
    bm25_k : int
        Số docs trả về từ BM25 retriever (mặc định 10).
    embedding_model : str
        Tên HuggingFace embedding model.
    chroma_path : str | None
        Đường dẫn lưu Chroma DB. Mặc định: ``<project>/chroma_db``.
    bm25_path : str | None
        Đường dẫn lưu BM25 pickle. Mặc định: ``<project>/bm25_retriever.pkl``.
    """
    if chroma_path is None:
        chroma_path = os.path.join(BASE_DIR, "chroma_db")
    if bm25_path is None:
        bm25_path = os.path.join(BASE_DIR, "bm25_retriever.pkl")

    # 1. Load & chunk
    print(f"📂 Đang quét thư mục: {folder_path}")
    chunks = load_and_chunk_folder(
        folder_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    if not chunks:
        print("❌ Không có chunk nào được tạo. Kiểm tra lại thư mục đầu vào.")
        return

    # 2. BM25
    print(f"\n📄 Đang tạo chỉ mục BM25 (k={bm25_k})...")
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = bm25_k

    os.makedirs(os.path.dirname(os.path.abspath(bm25_path)), exist_ok=True)
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25_retriever, f)
    print(f"  ✅ Đã lưu BM25 tại: {bm25_path}")

    # 3. Chroma
    print(f"\n🧠 Đang tải embedding model: {embedding_model}")
    embedding = HuggingFaceEmbeddings(model_name=embedding_model)

    print(f"💾 Đang tạo Chroma Vector DB tại: {chroma_path}")
    os.makedirs(os.path.dirname(os.path.abspath(chroma_path)), exist_ok=True)
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=chroma_path,
    )
    print(f"  ✅ Đã lưu Vector DB thành công ({len(chunks)} chunks)")

    return db


if __name__ == "__main__":
    folder = os.path.join(BASE_DIR, "data")
    print(f"🏁 Chạy create_vector_db với thư mục: {folder}")
    create_vector_db(folder)