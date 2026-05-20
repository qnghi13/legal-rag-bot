import os
import platform
import pytesseract
from pdf2image import convert_from_path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Cấu hình đường dẫn Tesseract (Nếu bạn dùng Windows)
# Hãy sửa đường dẫn này nếu bạn cài đặt Tesseract ở ổ đĩa khác
if platform.system() == "Windows":
    # Nếu code đang chạy trên máy tính Windows của bạn
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
else:
    # Nếu chạy trên Docker (Linux), pytesseract sẽ tự động tìm thấy tesseract trong PATH
    # Nên chúng ta không cần gán đường dẫn ở đây.
    pass

def extract_text_from_scanned_pdf(file_path):
    print(f"    🔍 Phát hiện PDF Scan, đang kích hoạt OCR (Tiếng Việt)...")
    text_content = ""
    
    # 1. Chuyển PDF thành danh sách các hình ảnh (mỗi trang là 1 ảnh)
    images = convert_from_path(file_path)
    
    # 2. Dùng OCR quét từng ảnh
    for i, image in enumerate(images):
        # Tham số lang='vie' là cực kỳ quan trọng để bắt tiếng Việt có dấu
        page_text = pytesseract.image_to_string(image, lang='vie')
        text_content += f"\n--- Trang {i+1} ---\n" + page_text
        
    # 3. Trả về dưới định dạng Document của Langchain
    return [Document(page_content=text_content, metadata={"source": file_path})]

def load_and_chunk_folder(folder_path):
    print(f"📂 Đang quét toàn bộ file trong thư mục: {folder_path}...")
    all_documents = []
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        try:
            if filename.endswith(".pdf"):
                print(f"  -> Đang xử lý PDF: {filename}")
                # Thử đọc bằng PyPDF (Dành cho file PDF Text thông thường)
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                
                # Logic kiểm tra: Nếu tổng số chữ của tất cả các trang < 50 chữ (tức là rỗng)
                # -> Đây 100% là PDF Scan!
                total_text = "".join([d.page_content for d in docs]).strip()
                if len(total_text) < 50:
                    docs = extract_text_from_scanned_pdf(file_path)
                
                all_documents.extend(docs)
                
            elif filename.endswith((".txt", ".md")):
                print(f"  -> Đang xử lý Text: {filename}")
                loader = TextLoader(file_path, encoding='utf-8')
                all_documents.extend(loader.load())
        except Exception as e:
            print(f"❌ Lỗi khi đọc file {filename}: {e}")

    print(f"\n✅ Đã bóc tách xong chữ từ tất cả các file.")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    chunks = text_splitter.split_documents(all_documents)
    print(f"🔪 Đã băm thành {len(chunks)} chunks nhỏ!")
    return chunks

if __name__ == "__main__":
    folder_path = "../data"
    load_and_chunk_folder(folder_path)