"""Evaluation script for legal-rag-bot — sử dụng Ragas để đánh giá RAG pipeline.

Cách dùng:
    python evaluate.py
    python evaluate.py --judge-llm llama-3.1-70b-versatile
    python evaluate.py --questions '["câu hỏi 1", "câu hỏi 2"]'
"""

import argparse
import os
import sys
from dotenv import load_dotenv
import pandas as pd
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.run_config import RunConfig

from src.bot import get_hr_bot
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# ---------------------------------------------------------------------------
# Test set mặc định
# ---------------------------------------------------------------------------
DEFAULT_QUESTIONS = [
    "Người lao động có phải trả chi phí tuyển dụng lao động không?",
    "Hợp đồng lao động có bắt buộc phải lập thành văn bản không?",
    "Người sử dụng lao động có được giữ bản chính văn bằng của người lao động không?",
    "Người lao động có thể ký nhiều hợp đồng lao động cùng lúc không?",
    "Thời giờ làm việc bình thường tối đa trong một tuần là bao nhiêu giờ?",
    "Giờ làm việc ban đêm được tính từ mấy giờ đến mấy giờ?",
    "Người lao động làm thêm giờ trong một tháng tối đa bao nhiêu giờ?",
    "Người lao động được nghỉ giữa giờ tối thiểu bao nhiêu phút khi làm việc ban ngày?",
    "Người lao động được nghỉ bao nhiêu ngày dịp Tết Âm lịch?",
    "Sau bao nhiêu năm làm việc thì người lao động được tăng thêm ngày nghỉ hằng năm?",
    
    # Các câu ngoài phạm vi / không có trong ngữ cảnh
    "Mức lương tối thiểu vùng năm 2026 là bao nhiêu?",
    "Người lao động nữ sinh con được nghỉ thai sản bao nhiêu tháng?",
    "Thuế thu nhập cá nhân được tính như thế nào?",
    "Doanh nghiệp nợ bảo hiểm xã hội sẽ bị phạt bao nhiêu tiền?"
]

DEFAULT_GROUND_TRUTHS = [
    "Không. Người lao động không phải trả chi phí cho việc tuyển dụng lao động.",
    "Có. Hợp đồng lao động phải được giao kết bằng văn bản, trừ trường hợp hợp đồng có thời hạn dưới 01 tháng có thể giao kết bằng lời nói.",
    "Không. Người sử dụng lao động không được giữ bản chính giấy tờ tùy thân, văn bằng, chứng chỉ của người lao động.",
    "Có. Người lao động có thể giao kết nhiều hợp đồng lao động với nhiều người sử dụng lao động nhưng phải bảo đảm thực hiện đầy đủ các nội dung đã giao kết.",
    "Thời giờ làm việc bình thường không quá 48 giờ trong 01 tuần.",
    "Giờ làm việc ban đêm được tính từ 22 giờ đến 06 giờ sáng ngày hôm sau.",
    "Số giờ làm thêm của người lao động không quá 40 giờ trong 01 tháng.",
    "Người lao động làm việc từ 06 giờ trở lên trong một ngày được nghỉ giữa giờ ít nhất 30 phút liên tục.",
    "Người lao động được nghỉ 05 ngày dịp Tết Âm lịch và hưởng nguyên lương.",
    "Cứ đủ 05 năm làm việc cho một người sử dụng lao động thì người lao động được tăng thêm 01 ngày nghỉ hằng năm.",

    # Ground truth cho câu ngoài phạm vi
    "Không trả lời được từ ngữ cảnh được cung cấp.",
    "Không trả lời được từ ngữ cảnh được cung cấp.",
    "Không trả lời được từ ngữ cảnh được cung cấp.",
    "Không trả lời được từ ngữ cảnh được cung cấp."
]


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Đánh giá RAG pipeline bằng Ragas metrics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Bot parameters
    parser.add_argument("--llm-model", default="llama-3.1-8b-instant",
                        help="Model Groq cho bot trả lời")
    parser.add_argument("--llm-temperature", type=float, default=0.2,
                        help="Temperature cho bot")
    parser.add_argument("--embedding-model", default="keepitreal/vietnamese-sbert",
                        help="Embedding model cho retrieval")
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-v2-m3",
                        help="CrossEncoder model cho re-ranking")
    parser.add_argument("--reranker-max-length", type=int, default=512,
                        help="Max length cho reranker")
    parser.add_argument("--retrieval-k", type=int, default=10,
                        help="Số docs từ dense retriever")
    parser.add_argument("--rerank-top-k", type=int, default=3,
                        help="Số docs giữ lại sau rerank")
    parser.add_argument("--chroma-path", default=None,
                        help="Đường dẫn Chroma DB")
    parser.add_argument("--bm25-path", default=None,
                        help="Đường dẫn BM25 pickle")

    # Judge parameters (riêng — không ảnh hưởng bot)
    parser.add_argument("--judge-llm", default="llama-3.3-70b-versatile",
                        help="Model Groq dùng làm giám khảo chấm điểm")
    parser.add_argument("--judge-embedding", default="keepitreal/vietnamese-sbert",
                        help="Embedding model cho giám khảo")
    parser.add_argument("--judge-temperature", type=float, default=0,
                        help="Temperature cho giám khảo (thường để 0 để đánh giá khách quan)")

    # Test data
    parser.add_argument("--questions", default=None,
                        help="JSON array string: '[\"q1\", \"q2\"]'")
    parser.add_argument("--ground-truths", default=None,
                        help="JSON array string: '[\"gt1\", \"gt2\"]'")

    # Output
    parser.add_argument("--output", default=None,
                        help="Đường dẫn file CSV để lưu kết quả")
    parser.add_argument("--verbose", action="store_true",
                        help="In chi tiết từng câu hỏi")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def run_evaluation(**kwargs):
    """Đánh giá RAG pipeline.

    Parameters
    ----------
    **kwargs : dict
        Có thể truyền bất kỳ tham số nào từ CLI để dễ dàng gọi từ code.
    """
    args = parse_args()
    # Override defaults với kwargs nếu có
    for key, value in kwargs.items():
        if hasattr(args, key) and value is not None:
            setattr(args, key, value)

    print("🚀 Bắt đầu quá trình đánh giá (Evaluation) hệ thống LegalBot...\n")

    # 1. Test set
    questions = args.questions
    if isinstance(questions, str):
        import json
        questions = json.loads(questions)
    if not questions:
        questions = DEFAULT_QUESTIONS

    ground_truths = args.ground_truths
    if isinstance(ground_truths, str):
        import json
        ground_truths = json.loads(ground_truths)
    if not ground_truths:
        ground_truths = DEFAULT_GROUND_TRUTHS

    if len(questions) != len(ground_truths):
        print(f"⚠️  Số lượng questions ({len(questions)}) khác ground_truths ({len(ground_truths)}).")

    print(f"📋 Số câu hỏi đánh giá: {len(questions)}")
    if args.verbose:
        for i, (q, gt) in enumerate(zip(questions, ground_truths), 1):
            print(f"  {i}. {q[:80]}...")

    # 2. Sinh câu trả lời từ Bot
    print(f"\n🤖 Khởi tạo Bot: model={args.llm_model}, "
          f"retrieval_k={args.retrieval_k}, rerank_top_k={args.rerank_top_k}")
    rag_chain = get_hr_bot(
        llm_model=args.llm_model,
        llm_temperature=args.llm_temperature,
        embedding_model=args.embedding_model,
        reranker_model=args.reranker_model,
        reranker_max_length=args.reranker_max_length,
        retrieval_k=args.retrieval_k,
        rerank_top_k=args.rerank_top_k,
        chroma_path=args.chroma_path,
        bm25_path=args.bm25_path,
        return_context_list=True,  # <--- Quan trọng: trả về list cho Ragas
    )

    print("\n🤖 Đang cho Bot làm bài thi...")
    answers = []
    contexts_list = []
    for i, q in enumerate(questions, 1):
        print(f"  [{i}/{len(questions)}] Hỏi: {q[:80]}")
        response = rag_chain.invoke({"input": q, "chat_history": []})

        answers.append(response["answer"])
        # Ragas yêu cầu contexts là list of list of strings
        # Mỗi câu hỏi → một list các context documents riêng lẻ
        contexts_list.append(response.get("context_list", []))

    # 3. Build dataset cho Ragas
    eval_data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths,
    }

    # Chỉ giữ các câu có ground truth đầy đủ
    valid_indices = [
        i for i, gt in enumerate(ground_truths)
        if gt and not gt.startswith("Không trả lời được")
    ]
    for key in eval_data:
        eval_data[key] = [eval_data[key][i] for i in valid_indices]

    if not eval_data["question"]:
        print("❌ Không có câu hỏi hợp lệ để đánh giá. Hãy cập nhật ground_truths.")
        return

    dataset = Dataset.from_dict(eval_data)

    # 4. Giám khảo AI
    print(f"\n⚖️ Giám khảo AI: model={args.judge_llm}, embedding={args.judge_embedding}")
    judge_llm = ChatGroq(
        model=args.judge_llm,
        temperature=args.judge_temperature,
    )
    judge_embedding = HuggingFaceEmbeddings(model_name=args.judge_embedding)

    config = RunConfig(timeout=120, max_workers=2, max_retries=10)

    # 5. Chạy Ragas
    print("  Đang chấm điểm...")
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
        run_config=config,
        raise_exceptions=False,
    )

    # 6. Kết quả
    print("\n" + "=" * 55)
    print("📊 BẢNG ĐIỂM ĐÁNH GIÁ (0 → 1) — càng cao càng tốt:")
    print("=" * 55)
    df_result = result.to_pandas()

    metrics_map = {
        "faithfulness": "Bot không bịa chuyện",
        "answer_relevancy": "Trả lời đúng trọng tâm",
        "context_precision": "Re-ranker lọc tốt",
        "context_recall": "DB tìm đủ ý",
    }

    for col, desc in metrics_map.items():
        if col in df_result.columns:
            print(f"  - {desc:30s}: {df_result[col].mean():.4f}")

    print(f"\n  - Số câu đánh giá: {len(df_result)}")
    print(f"  - Cấu hình: retrieval_k={args.retrieval_k}, "
          f"rerank_top_k={args.rerank_top_k}, "
          f"model={args.llm_model}")

    # 7. Lưu file nếu cần
    if args.output:
        df_result.to_csv(args.output, index=False, encoding="utf-8-sig")
        print(f"\n✅ Đã lưu kết quả chi tiết vào: {args.output}")

    # 8. In chi tiết từng câu nếu --verbose
    if args.verbose:
        print("\n" + "-" * 55)
        print("📋 CHI TIẾT TỪNG CÂU:")
        print("-" * 55)
        for i, row in df_result.iterrows():
            print(f"\n  Q{i + 1}: {row.get('question', '')[:100]}")
            print(f"      faithfulness={row.get('faithfulness', 'N/A')} "
                  f"| answer_relevancy={row.get('answer_relevancy', 'N/A')}")
            if "context_precision" in row:
                print(f"      context_precision={row.get('context_precision', 'N/A')} "
                      f"| context_recall={row.get('context_recall', 'N/A')}")

    return df_result


if __name__ == "__main__":
    run_evaluation()