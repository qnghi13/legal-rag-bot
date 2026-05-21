import os
import re
import platform
import pytesseract
import pymupdf4llm  # Thư viện mới trích xuất PDF thẳng ra Markdown
from pdf2image import convert_from_path
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# 1. Cấu hình Tesseract OCR (Giữ nguyên)
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text_from_scanned_pdf(file_path):
    print(f"    🔍 Kích hoạt OCR cho file Scan...")
    
    # Tạo pseudo-markdown (Markdown giả) để Splitter vẫn nhận diện được
    filename = os.path.basename(file_path)
    text_content = f"# Tài liệu Scan: {filename}\n\n"
    
    images = convert_from_path(file_path)
    for i, image in enumerate(images):
        page_text = pytesseract.image_to_string(image, lang='vie')
        text_content += f"## Trang {i+1}\n\n{page_text}\n\n"
        
    return text_content

def normalize_legal_headings(text):
    """
    Dọn dẹp rác markdown của thư viện cũ và dùng Regex nhận diện cấu trúc Luật VN.
    Quy tắc:
    # -> Chương (H1)
    ## -> Mục (H2)
    ### -> Điều (H3)
    """
    # 1. Xóa rác Markdown do pymupdf4llm tạo ra (Xóa các dấu ## **, **, __)
    text = re.sub(r'#+\s*\*\*', '', text)
    text = re.sub(r'#+\s*', '', text)
    text = re.sub(r'\*\*', '', text)
    
    # 2. Gắn lại thẻ Markdown chuẩn xác dựa vào Keyword của Luật
    # Bắt chữ "Chương" + số La Mã ở đầu dòng -> Gắn thẻ H1 (#)
    text = re.sub(r'(?m)^(Chương\s+[IVXLCDM]+.*?)\s*[\r\n]+\s*(?!(?:Điều|Mục|Chương))(.+)', r'# \1 - \2', text)
    
    # Bắt chữ "Mục" + số ở đầu dòng -> Gắn thẻ H2 (##)
    text = re.sub(r'(?m)^(Mục\s+\d+.*)', r'## \1', text)
    
    # Bắt chữ "Điều" + số ở đầu dòng -> Gắn thẻ H3 (###)
    text = re.sub(r'(?m)^(Điều\s+\d+\..*)', r'### \1', text)
    
    return text

def process_file_to_markdown(file_path):
    """Hàm này chịu trách nhiệm biến mọi thể loại file thành 1 chuỗi Markdown thuần"""
    filename = os.path.basename(file_path)
    
    if filename.endswith(".pdf"):
        # Thử dùng PyMuPDF4LLM để convert thẳng PDF sang Markdown
        md_text = pymupdf4llm.to_markdown(file_path)
        normalized_md = normalize_legal_headings(md_text)
        
        # # Kiểm tra file Scan: Nếu văn bản trả về quá ngắn, dùng OCR
        # if len(md_text.strip()) < 50:
        #     md_text = extract_text_from_scanned_pdf(file_path)
        return normalized_md
        
    elif filename.endswith((".txt", ".md")):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
            
    return ""

def load_and_chunk_folder(folder_path):
    print(f"📂 Đang quét toàn bộ file trong thư mục: {folder_path}...")
    final_chunks = []
    
    # 2. CẤU HÌNH TÁCH MARKDOWN (SEMANTIC CHUNKING)
    headers_to_split_on = [
        ("#", "Chuong"),
        ("##", "Muc"),
        ("###", "Dieu"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    
    # 3. CẤU HÌNH TÁCH TEXT CHO CÁC ĐOẠN QUÁ DÀI
    # Sau khi cắt theo Heading, nếu có 1 chương dài 5000 chữ, ta phải cắt nhỏ nó ra tiếp
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    # Tiến hành xử lý từng file
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        try:
            print(f"  -> Đang xử lý: {filename}")
            
            # Bước A: Chuyển file thành chuỗi Markdown
            raw_markdown = process_file_to_markdown(file_path)
            if not raw_markdown:
                continue
                
            # Bước B: Tách theo Heading (Giữ lại Metadata)
            md_docs = markdown_splitter.split_text(raw_markdown)
            
            # Bước C: Ép thêm source (nguồn file) vào metadata của từng đoạn
            for doc in md_docs:
                doc.metadata["source"] = filename
                
            # Bước D: Tách nhỏ các đoạn văn còn quá dài
            chunks = text_splitter.split_documents(md_docs)
            for chunk in chunks:
                chuong = chunk.metadata.get("Chuong", "")
                muc = chunk.metadata.get("Muc", "")
                dieu = chunk.metadata.get("Dieu", "")
                
                # Tạo một "Bảng tên" cho Chunk
                header_tag = f"[{chuong} | {muc} | {dieu}]".replace(" |  | ", " | ").strip(" | ")
                
                # Dán Bảng tên lên ĐẦU đoạn nội dung
                chunk.page_content = f"{header_tag}\n{chunk.page_content}"

            final_chunks.extend(chunks)
            
        except Exception as e:
            print(f"❌ Lỗi khi đọc file {filename}: {e}")

    print(f"\n✅ Quá trình băm hoàn tất!")
    print(f"🔪 Đã tạo ra {len(final_chunks)} Semantic Chunks (có chứa Metadata Heading)!")
    
    # In thử 1 chunk ra xem Metadata nó xịn cỡ nào
    if final_chunks:
        print("\n--- [DEBUG] Mẫu 1 Chunk sau khi cắt ---")
        print(f"Metadata: {final_chunks[0].metadata}")
        print(f"Nội dung: {final_chunks[0].page_content[:1000]}...")
        
    return final_chunks

if __name__ == "__main__":
    folder_path = "../data"
    load_and_chunk_folder(folder_path)