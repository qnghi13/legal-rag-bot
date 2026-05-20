import os
from dotenv import load_dotenv
import pandas as pd
from datasets import Dataset

# Import Ragas
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

# Import Bot của bạn
from src.bot import get_hr_bot
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

def run_evaluation():
    print("🚀 Bắt đầu quá trình đánh giá (Evaluation) hệ thống LegalBot...")
    
    # 1. Chuẩn bị Testset (Tập dữ liệu kiểm thử - Ground Truth)
    # Đây là những câu hỏi và câu trả lời CHUẨN do chính bạn viết ra từ file PDF.
    eval_data = {
        "question": [
            "Người lao động có được làm thêm giờ quá 40 tiếng 1 tháng không?",
            "Thử việc tối đa bao nhiêu ngày đối với công việc cần trình độ đại học?",
            "Công ty có quyền giữ bản chính bằng đại học của tôi khi nhận việc không?"
        ],
        "ground_truth": [ # Câu trả lời đúng chuẩn của con người
            "Không. Theo Bộ luật Lao động, thời gian làm thêm giờ không được quá 40 giờ trong 01 tháng.",
            "Thời gian thử việc không quá 60 ngày đối với công việc có chức danh nghề nghiệp cần trình độ chuyên môn, kỹ thuật từ cao đẳng trở lên.",
            "Không. Người sử dụng lao động không được giữ bản chính giấy tờ tùy thân, văn bằng, chứng chỉ của người lao động."
        ]
    }
    
    # 2. Sinh câu trả lời từ Bot của bạn
    rag_chain = get_hr_bot()
    answers = []
    contexts = []
    
    print("\n🤖 Đang cho Bot làm bài thi...")
    for q in eval_data["question"]:
        print(f"Hỏi: {q}")
        response = rag_chain.invoke({"input": q, "chat_history": []})
        
        answers.append(response["answer"])
        # Ragas yêu cầu contexts phải là một list các chuỗi
        contexts.append([response["context"]]) 
        
    eval_data["answer"] = answers
    eval_data["contexts"] = contexts
    
    # Chuyển đổi thành định dạng Dataset của HuggingFace
    dataset = Dataset.from_dict(eval_data)
    
    # 3. Khởi tạo LLM và Embedding để làm "Giám khảo"
    # Dùng Groq và model Sbert hiện tại của bạn để làm giám khảo luôn cho tiết kiệm
    print("\n⚖️ Giám khảo AI đang chấm điểm...")
    judge_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    judge_embedding = HuggingFaceEmbeddings(model_name="keepitreal/vietnamese-sbert")
    
    # 4. Chạy hàm đánh giá
    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=judge_llm,
        embeddings=judge_embedding,
        raise_exceptions=False
    )
    
    # 5. In kết quả ra màn hình
    print("\n" + "="*50)
    print("📊 BẢNG ĐIỂM ĐÁNH GIÁ (0 -> 1):")
    print("="*50)
    df_result = result.to_pandas()
    
    # Hiển thị điểm trung bình
    print(f"- Faithfulness (Bot không bịa chuyện): {df_result['faithfulness'].mean():.2f}")
    print(f"- Answer Relevance (Trả lời đúng trọng tâm): {df_result['answer_relevancy'].mean():.2f}")
    print(f"- Context Precision (Re-ranker lọc tốt): {df_result['context_precision'].mean():.2f}")
    print(f"- Context Recall (DB tìm đủ ý): {df_result['context_recall'].mean():.2f}")
    
    # Lưu ra file Excel để bỏ vào Portfolio
    # df_result.to_csv("rag_evaluation_report.csv", index=False, encoding='utf-8-sig')
    # print("\n✅ Đã lưu kết quả chi tiết vào file 'rag_evaluation_report.csv'")

if __name__ == "__main__":
    run_evaluation()