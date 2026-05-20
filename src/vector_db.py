import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ĐỔI TÊN HÀM IMPORT Ở ĐÂY
from ingest import load_and_chunk_folder 

load_dotenv()
CHROMA_PATH = "../chroma_db"

# ĐỔI TÊN BIẾN ĐẦU VÀO TỪ file_path THÀNH folder_path
def create_vector_db(folder_path):
    # Gọi hàm quét thư mục
    chunks = load_and_chunk_folder(folder_path)
    
    print("\nĐang tải mô hình Embedding (BAAI/bge-m3)...")
    embedding_model = HuggingFaceEmbeddings(model_name="keepitreal/vietnamese-sbert")
    
    print("Đang tạo Vector Database và lưu xuống ổ cứng...")
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_PATH
    )
    print(f"✅ Đã lưu Vector DB thành công tại thư mục: {CHROMA_PATH}")
    return db

if __name__ == "__main__":
    # CHỈ ĐỊNH ĐƯỜNG DẪN LÀ CẢ THƯ MỤC DATA
    folder_path = "../data" 
    create_vector_db(folder_path)