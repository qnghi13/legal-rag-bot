import pymupdf4llm

md = pymupdf4llm.to_markdown("../data/luat_lao_dong.pdf")
# xuất file markdown ra ổ cứng để kiểm tra
with open("../data/luat_lao_dong.md", "w", encoding="utf-8") as f:
    f.write(md)