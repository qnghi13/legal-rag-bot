from mineru import MinerU

mineru = MinerU()

result = mineru.parse("../data/luat_lao_dong.pdf")

print(result.markdown)