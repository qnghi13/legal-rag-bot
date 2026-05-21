import os
import re
import pickle
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# Import trực tiếp thư viện gốc, KHÔNG thông qua LangChain nữa
from sentence_transformers import CrossEncoder

load_dotenv()
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
BM25_PATH = os.path.join(BASE_DIR, "bm25_retriever.pkl")

# Load BM25 Retriever
with open(BM25_PATH, "rb") as f:
    bm25_retriever = pickle.load(f)

def get_hr_bot():
    print("Đang khởi động LLM...")
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)
    
    print("Đang kết nối Vector DB...")
    embedding_model = HuggingFaceEmbeddings(model_name="keepitreal/vietnamese-sbert")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_model)
    
    # 1. Base Retriever: Lấy 10 kết quả (Mở rộng lưới để bắt cá)
    base_retriever = db.as_retriever(search_kwargs={"k": 10})
    
    # 2. Load Model Re-ranker trực tiếp từ HuggingFace (Chạy offline)
    print("Đang tải Giáo sư Re-ranker (BAAI/bge-reranker-v2-m3)...")
    reranker_model = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)

    # ==========================================
    # 🌟 CUSTOM FUNCTION: TỰ VIẾT RE-RANKER 
    # ==========================================
    def retrieve_and_rerank(query: str) -> str:
        # 1. Thu thập từ 2 nguồn (Dense + Sparse)
        dense_docs = base_retriever.invoke(query)
        sparse_docs = bm25_retriever.invoke(query)
        all_docs = dense_docs + sparse_docs
        
        # 2. Lọc trùng lặp (Deduplication) dựa trên nội dung
        unique_docs = []
        seen_content = set()
        for doc in all_docs:
            if doc.page_content not in seen_content:
                unique_docs.append(doc)
                seen_content.add(doc.page_content)
                
        if not unique_docs: return ""

        # 3. Đưa cho Giáo sư chấm điểm
        pairs = [[query, doc.page_content] for doc in unique_docs]
        scores = reranker_model.predict(pairs)
        
        scored_docs = list(zip(unique_docs, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        top_3_docs = [doc for doc, score in scored_docs[:3]]
        
        # 4. Gắn Metadata (Chương, Điều) vào Text
        formatted_contexts = []
        for i, doc in enumerate(top_3_docs):
            meta = doc.metadata
            chuong = meta.get("Chuong", "")
            muc = meta.get("Muc", "")
            dieu = meta.get("Dieu", "")
            source = meta.get("source", "Tài liệu nội bộ")
            
            headers = [h for h in [chuong, muc, dieu] if h]
            hierarchy = " > ".join(headers) if headers else "Nội dung chung"
            
            chunk_str = f"--- TÀI LIỆU SỐ {i+1} ---\n"
            chunk_str += f"[Nguồn]: {source}\n"
            chunk_str += f"[Nội dung]: {doc.page_content}"
            
            formatted_contexts.append(chunk_str)
            
        return "\n\n".join(formatted_contexts)
    # ==========================================

    # Kịch bản 1: Viết lại câu hỏi
    rephrase_prompt = ChatPromptTemplate.from_messages([
        ("system", """Bạn là một chuyên gia phân tích ngôn ngữ pháp lý.
                    Nhiệm vụ của bạn là đọc lịch sử trò chuyện và câu hỏi mới của người dùng.
                    Hãy viết lại câu hỏi này thành một CÂU HỎI ĐỘC LẬP (Standalone question) rõ ràng, mang đầy đủ ngữ cảnh pháp lý để hệ thống có thể tra cứu Luật.
                    - Nếu câu hỏi có chứa đại từ nhân xưng (tôi, anh ấy, sếp, công ty), hãy giữ nguyên hoặc làm rõ dựa trên lịch sử.
                    - KHÔNG TRẢ LỜI CÂU HỎI, chỉ viết lại câu hỏi. Nếu câu hỏi đã đủ ý, hãy giữ nguyên."""),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    # Kịch bản 2: Trả lời
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", """Bạn là một Luật sư Tư vấn Luật Lao động Việt Nam ảo, chuyên nghiệp, khách quan và đáng tin cậy.
                    Nhiệm vụ của bạn là tư vấn pháp lý cho người dùng CHỈ DỰA VÀO CÁC ĐIỀU LUẬT (NGỮ CẢNH) DƯỚI ĐÂY.

                    CÁC NGUYÊN TẮC TỐI THƯỢNG (VI PHẠM SẼ BỊ PHẠT):
                    1. CHỈ sử dụng thông tin trong phần "Ngữ cảnh". TUYỆT ĐỐI KHÔNG sử dụng kiến thức bên ngoài của bạn, KHÔNG tự bịa ra các Điều, Khoản luật.
                    2. Nếu Ngữ cảnh cung cấp không chứa thông tin để trả lời câu hỏi, HÃY TRẢ LỜI CHÍNH XÁC CÂU NÀY: "Xin lỗi, dựa trên dữ liệu luật pháp tôi đang có, tôi không tìm thấy quy định cụ thể về vấn đề bạn đang hỏi."
                    3. Khi trả lời, NẾU CÓ THỂ, hãy trích dẫn cụ thể tên Điều luật (VD: "Theo Điều 34...").
                    4. Hãy suy luận từng bước. Đọc kỹ câu hỏi, đối chiếu với Ngữ cảnh, tóm tắt ý chính và trả lời một cách lịch sự, mạch lạc.
                    5. Nếu câu hỏi là những câu giao tiếp thông thường (xin chào, cảm ơn), hãy đáp lại lịch sự nhưng nhanh chóng hướng người dùng về chủ đề tra cứu luật.

                    Ngữ cảnh (Trích xuất từ văn bản Luật):
        ---
        {context}
        ---"""),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    # Ráp toàn bộ đường ống LCEL
    retrieval_chain = RunnablePassthrough.assign(
        context=rephrase_prompt | llm | StrOutputParser() | RunnableLambda(retrieve_and_rerank)
    )

    # BƯỚC 2: Ráp thêm phần trả lời (Generation), nhưng GIỮ LẠI ngữ cảnh
    rag_chain = retrieval_chain | RunnablePassthrough.assign(
        answer=qa_prompt | llm | StrOutputParser()
    )

    return rag_chain