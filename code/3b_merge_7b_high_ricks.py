# 文件路径配置
file1_path = "/root/autodl-tmp/project/output/3b_fire_risk.txt"  # qwen2.5-vl-3b-sft-high-risk预测的高风险标签
file2_path = "/root/autodl-tmp/project/output/7b_fire_risk.txt"  # 多标签分类模型qwen2.5-vl-7b-sft模型预测标签
output_path = "/root/autodl-tmp/project/output/b-submit.txt"  # 输出结果文件

# 读取 file1.txt 中的文件名（高风险图片）
high_risk_files = set()
with open(file1_path, "r", encoding="utf-8") as f1:
    for line in f1:
        parts = line.strip().split()
        if len(parts) >= 2 and parts[1] == "高风险":
            high_risk_files.add(parts[0])

# 遍历 file2.txt 并更新标签（只对出现在 file1.txt 的文件名处理）
with open(file2_path, "r", encoding="utf-8") as f2, \
     open(output_path, "w", encoding="utf-8") as fout:
    for line in f2:
        parts = line.strip().split()
        if len(parts) >= 2:
            filename, label = parts[0], parts[1]
            if filename in high_risk_files and label != "高风险":
                fout.write(f"{filename}\t高风险\n")
            else:
                fout.write(line)
        else:
            fout.write(line)  # 写入格式异常的行

print("匹配更新完成，输出文件已保存为 b-submit.txt")
