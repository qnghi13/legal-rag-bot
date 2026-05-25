import os
import re
import pickle
import functools
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder

load_dotenv()
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Module-level model cache — tránh load lại model nhiều lần
# ---------------------------------------------------------------------------
_MODEL_CACHE: dict[str, object] = {}


def _get_embedding_model(model_name: str = "keepitreal/vietnamese-sbert"):
    key = f"embed:{model_name}"
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = HuggingFaceEmbeddings(model_name=model_name)
    return _MODEL_CACHE[key]


def _get_reranker(model_name: str = "BAAI/bge-reranker-v2-m3", max_length: int = 512):
    key = f"rerank:{model_name}:{max_length}"
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = CrossEncoder(model_name, max_length=max_length)
    return _MODEL_CACHE[key]


def _get_bm25_retriever(bm25_path: str | None = None):
    """Lazy-load BM25 retriever — chỉ load khi thực sự cần."""
    if bm25_path is None:
        bm25_path = os.path.join(BASE_DIR, "bm25_retriever.pkl")
    key = f"bm25:{bm25_path}"
    if key not in _MODEL_CACHE:
        if not os.path.exists(bm25_path):
            raise FileNotFoundError(
                f"Không tìm thấy file BM25 tại: {bm25_path}\n"
                "Hãy chạy `vector_db.py` trước để tạo chỉ mục."
            )
        with open(bm25_path, "rb") as f:
            _MODEL_CACHE[key] = pickle.load(f)
    return _MODEL_CACHE[key]


# ---------------------------------------------------------------------------
# Core RAG function
# ---------------------------------------------------------------------------
def get_hr_bot(
    *,
    # LLM
    llm_model: str = "llama-3.1-8b-instant",
    llm_temperature: float = 0.2,
    # Embedding
    embedding_model: str = "keepitreal/vietnamese-sbert",
    # Reranker
    reranker_model: str = "BAAI/bge-reranker-v2-m3",
    reranker_max_length: int = 512,
    # Retrieval
    retrieval_k: int = 10,  # số docs query từ dense retriever
    rerank_top_k: int = 3,  # số docs giữ lại sau rerank
    # Paths
    chroma_path: str | None = None,
    bm25_path: str | None = None,
    # Output
    return_context_list: bool = False,
):
    """Build RAG chain với các tham số có thể tuỳ chỉnh để dễ evaluation.

    Parameters
    ----------
    llm_model : str
        Tên model Groq (mặc định "llama-3.1-8b-instant").
    llm_temperature : float
        Temperature cho LLM (mặc định 0.2).
    embedding_model : str
        Tên HuggingFace embeddings model.
    reranker_model : str
        Tên CrossEncoder model cho re-ranking.
    reranker_max_length : int
        Max length cho reranker tokenizer.
    retrieval_k : int
        Số documents lấy từ dense retriever.
    rerank_top_k : int
        Số documents giữ lại sau re-rank.
    chroma_path : str | None
        Đường dẫn Chroma DB. Mặc định: ``<project>/chroma_db``.
    bm25_path : str | None
        Đường dẫn BM25 pickle. Mặc định: ``<project>/bm25_retriever.pkl``.
    return_context_list : bool
        Nếu True, chain trả về thêm key ``context_list`` (list[str])
        để dùng cho Ragas evaluation.

    Returns
    -------
    LCEL Runnable trả về dict với keys ``answer`` và ``context``
    (và ``context_list`` nếu ``return_context_list=True``).
    """

    # ---------- Khởi tạo các model (đã cache sẵn) ----------
    if chroma_path is None:
        chroma_path = os.path.join(BASE_DIR, "chroma_db")

    print(f"  [bot] LLM: {llm_model} (temp={llm_temperature})")
    llm = ChatGroq(model=llm_model, temperature=llm_temperature)

    print(f"  [bot] Embedding: {embedding_model}")
    embedding = _get_embedding_model(embedding_model)

    print(f"  [bot] Kết nối Vector DB: {chroma_path}")
    db = Chroma(persist_directory=chroma_path, embedding_function=embedding)

    print(f"  [bot] Retriever: k={retrieval_k}")
    base_retriever = db.as_retriever(search_kwargs={"k": retrieval_k})

    print(f"  [bot] Reranker: {reranker_model}")
    reranker = _get_reranker(reranker_model, max_length=reranker_max_length)

    print(f"  [bot] BM25 (lazy)")
    bm25_retriever = _get_bm25_retriever(bm25_path)

    # ---------- Re-ranker logic ----------
    def _retrieve_docs(query: str):
        """Internal: retrieve, dedup, rerank, return (top_docs, formatted_string)."""
        dense_docs = base_retriever.invoke(query)
        sparse_docs = bm25_retriever.invoke(query)
        all_docs = dense_docs + sparse_docs

        # Dedup bằng hash để tránh tốn bộ nhớ
        seen = set()
        unique_docs = []
        for doc in all_docs:
            h = hash(doc.page_content)
            if h not in seen:
                seen.add(h)
                unique_docs.append(doc)

        if not unique_docs:
            return [], ""

        # Rerank
        pairs = [[query, doc.page_content] for doc in unique_docs]
        scores = reranker.predict(pairs)

        scored = list(zip(unique_docs, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        return scored[:rerank_top_k]

    def retrieve_and_rerank(query: str) -> dict:
        """Trả về dict để merge vào chain state (context + tùy chọn context_list)."""
        top = _retrieve_docs(query)
        if not top:
            return {"context": ""} | (
                {"context_list": []} if return_context_list else {}
            )

        top_docs = [doc for doc, _ in top]

        # Format thành string (dùng cho LLM)
        parts = []
        doc_texts = []
        for i, doc in enumerate(top_docs):
            meta = doc.metadata
            chuong = meta.get("Chuong", "")
            muc = meta.get("Muc", "")
            dieu = meta.get("Dieu", "")
            source = meta.get("source", "Tài liệu nội bộ")

            headers = [h for h in [chuong, muc, dieu] if h]
            hierarchy = " > ".join(headers) or "Nội dung chung"

            formatted = (
                f"--- TÀI LIỆU SỐ {i + 1} ---\n"
                f"[Nguồn]: {source}\n"
                f"[Nội dung]: {doc.page_content}"
            )
            parts.append(formatted)
            doc_texts.append(formatted)

        result = {"context": "\n\n".join(parts)}
        if return_context_list:
            result["context_list"] = doc_texts
        return result

    # ---------- Prompts ----------
    rephrase_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Bạn là một chuyên gia phân tích ngôn ngữ pháp lý.\n"
            "Nhiệm vụ của bạn là đọc lịch sử trò chuyện và câu hỏi mới của người dùng.\n"
            "Hãy viết lại câu hỏi này thành một CÂU HỎI ĐỘC LẬP (Standalone question) rõ ràng, "
            "mang đầy đủ ngữ cảnh pháp lý để hệ thống có thể tra cứu Luật.\n"
            "- Nếu câu hỏi có chứa đại từ nhân xưng (tôi, anh ấy, sếp, công ty), "
            "hãy giữ nguyên hoặc làm rõ dựa trên lịch sử.\n"
            "- KHÔNG TRẢ LỜI CÂU HỎI, chỉ viết lại câu hỏi. Nếu câu hỏi đã đủ ý, hãy giữ nguyên."
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    qa_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Bạn là một Luật sư Tư vấn Luật Lao động Việt Nam ảo, chuyên nghiệp, khách quan và đáng tin cậy.\n"
            "Nhiệm vụ của bạn là tư vấn pháp lý cho người dùng CHỈ DỰA VÀO CÁC ĐIỀU LUẬT (NGỮ CẢNH) DƯỚI ĐÂY.\n\n"
            "CÁC NGUYÊN TẮC TỐI THƯỢNG (VI PHẠM SẼ BỊ PHẠT):\n"
            "1. CHỈ sử dụng thông tin trong phần \"Ngữ cảnh\". "
            "TUYỆT ĐỐI KHÔNG sử dụng kiến thức bên ngoài của bạn, "
            "KHÔNG tự bịa ra các Điều, Khoản luật.\n"
            "2. Nếu Ngữ cảnh cung cấp không chứa thông tin để trả lời câu hỏi, "
            "HÃY TRẢ LỜI CHÍNH XÁC CÂU NÀY: "
            "\"Xin lỗi, dựa trên dữ liệu luật pháp tôi đang có, "
            "tôi không tìm thấy quy định cụ thể về vấn đề bạn đang hỏi.\"\n"
            "3. Khi trả lời, NẾU CÓ THỂ, hãy trích dẫn cụ thể tên Điều luật "
            "(VD: \"Theo Điều 34...\").\n"
            "4. Hãy suy luận từng bước. Đọc kỹ câu hỏi, đối chiếu với Ngữ cảnh, tóm tắt ý chính và trả lời một cách lịch sự, mạch lạc.\n"
            "5. Nếu câu hỏi là những câu giao tiếp thông thường (xin chào, cảm ơn), "
            "hãy đáp lại lịch sự nhưng nhanh chóng hướng người dùng về chủ đề tra cứu luật.\n\n"
            "Ngữ cảnh (Trích xuất từ văn bản Luật):\n---\n{context}\n---"
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    # ---------- LCEL pipeline ----------
    # Bước 1: viết lại câu hỏi
    rephrase_step = RunnablePassthrough.assign(
        _rephrased=rephrase_prompt | llm | StrOutputParser()
    )

    # Bước 2: retrieval + rerank, merge kết quả vào state
    retrieval_step = RunnableLambda(
        lambda state: state | retrieve_and_rerank(state["_rephrased"])
    )

    # Bước 3: generation (có kiểm tra context rỗng)
    rag_chain = rephrase_step | retrieval_step | RunnablePassthrough.assign(
        answer=_build_answer_step(qa_prompt, llm)
    )

    return rag_chain


# ---------------------------------------------------------------------------
# Answer step — kiểm tra context rỗng trước khi gọi LLM
# ---------------------------------------------------------------------------
def _build_answer_step(prompt, llm):
    """Nếu context rỗng, bỏ qua LLM — trả message mặc định luôn."""

    def _invoke(state: dict) -> str:
        context = (state.get("context") or "").strip()
        if not context:
            return {
                "Xin lỗi, dựa trên dữ liệu luật pháp tôi đang có, tôi không tìm thấy quy định cụ thể về vấn đề bạn đang hỏi."  
            }
        # Context có -> gọi LLM bình thường
        chain = prompt | llm | StrOutputParser()
        return chain.invoke(state)

    return RunnableLambda(_invoke)