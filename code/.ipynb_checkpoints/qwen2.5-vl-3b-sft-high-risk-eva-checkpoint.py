from QwenModel import load_model, get_model_output
from qwen_vl_utils import process_vision_info
import argparse
from tqdm import tqdm
import os
from datetime import datetime
import torch

# 设置环境变量
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', default='/root/autodl-tmp/project/code/prompt/3b_high_risk.txt', type=str)  # 消防风险提示语
    parser.add_argument('--image_dir', default='/root/B_test/B', type=str)       # 图片文件夹路径
    parser.add_argument('--image_list', default='/root/B_test/B_new.txt', type=str)  # 只包含图片文件名的txt
    # 获取当前时间作为文件名后缀
    #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #default_output = f'./output/fire_risk_result_{timestamp}.txt'
    parser.add_argument('--output', default='./output/3b_fire_risk.txt', type=str)  # 输出预测结果
    parser.add_argument('--model_path', default="/root/autodl-tmp/project/model/llyllylly/Qwen2.5-VL-3B-Instruct-aug-12e", type=str)
    return parser.parse_args()

def process():
    configs = argparser()

    # 加载模型
    model, processor = load_model(configs.model_path)

    # 读取 prompt 内容
    with open(configs.prompt, "r") as f:
        prompt = f.read()

    # 从txt读取文件名（顺序会保留）
    with open(configs.image_list, "r") as f:
        image_names = [line.strip() for line in f if line.strip()]
        
    # 确保输出文件夹存在
    output_dir = os.path.dirname(configs.output)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 打开输出文件写入预测结果
    with open(configs.output, 'w') as fout:
        for image_name in tqdm(image_names, desc="Predicting"):
            image_path = os.path.join(configs.image_dir, image_name)
            if not os.path.exists(image_path):
                print(f"[Warning] File not found: {image_path}")
                continue
            # with torch.no_grad():  # 禁用梯度计算，减少显存占用
            #     result = get_model_output(prompt, image_path, model, processor)
            result = get_model_output(prompt, image_path, model, processor)
            predicted_label = result.strip()
            #print("image_path:",image_path)
            #print("predicted_label:",predicted_label)
            fout.write(f"{image_name}\t{predicted_label}\n")

if __name__ == "__main__":
    process()
