import random

def shuffle_file(input_file, output_file):
    # 读取原始文件内容
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 随机打乱行顺序
    random.shuffle(lines)
    
    # 写入新文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print(f"文件已随机打乱并保存到 {output_file}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("用法: python shuffle_file.py 输入文件.txt 输出文件.txt")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    shuffle_file(input_file, output_file)

# 用于A榜数据集train标签随机打乱，增加鲁棒性和适应性（数据预处理） python shuffle_file.py /root/images/train.txt /root/images/shuffle_train.txt