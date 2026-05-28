"""Default evaluation datasets."""

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
    "Mức lương tối thiểu vùng năm 2026 là bao nhiêu?",
    "Người lao động nữ sinh con được nghỉ thai sản bao nhiêu tháng?",
    "Thuế thu nhập cá nhân được tính như thế nào?",
    "Doanh nghiệp nợ bảo hiểm xã hội sẽ bị phạt bao nhiêu tiền?",
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
    "Không trả lời được từ ngữ cảnh được cung cấp.",
    "Không trả lời được từ ngữ cảnh được cung cấp.",
    "Không trả lời được từ ngữ cảnh được cung cấp.",
    "Không trả lời được từ ngữ cảnh được cung cấp.",
]
