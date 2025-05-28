import json
import os




if __name__ == "__main__":
    # 路径配置

    image_folder_path = r"/root/autodl-tmp/project/LLaMA-Factory/data/train-aug"  # 图像文件夹路径
    label_file_path = r"/root/images/train_aug.txt"    # 你的标签文件路径
    output_json_path = "/root/autodl-tmp/project/LLaMA-Factory/data/train-aug.json"  # 输出的 JSON 文件名

    # 读取标签
    data_list = []
    with open(label_file_path, "r", encoding="utf-8") as f:
        for line in f:
            image_name, label = line.strip().split("\t")

            # 构造对话条目
            item = {
                "messages": [
                    {
                        "content": "<image>你是一位专注于图像安全风险分析的消防隐患识别专家，擅长从日常拍摄图片中准确识别楼道内的安全隐患等级。你的任务是根据提供的图片，判断其是否为楼道场景，并识别其中是否存在消防安全隐患，以及隐患的风险等级。分类目标（只输出以下五类中的一个）：请从下列类别中选择最符合的一类：高风险：楼道中存在明显的起火隐患，满足以下任一条件：1）过道中停放电动车，注意有些图片角落里可能会有电动车。2）楼道中飞线充电。3）电瓶正在充电。中风险：楼道中存在大量堆积物或堆放大量纸箱、木质家具等能造成火势蔓延的堵塞物，并满足以下任一条件：1）楼道内堆积众多堆积物已经严重影响通行。2）楼道的堆积物中有明显可见的纸箱、木质或布质的家具、泡沫箱等可燃物品。低风险：楼道内有堆物现象但不严重，具体表现为楼道中有物品摆放但基本不影响通行，数量较少或靠边有序摆放。无风险：楼道干净整洁，无堆物、无杂物，通行无障碍。非楼道：图像内容与楼道无关，例如室外、教室、宿舍、商铺、仓库等。识别规则：1. 主导风险优先：如图中同时出现中等和高风险因素，按更高风险判断。2. 仅根据图像内容判断，不基于猜测或外部信息。3. 必须分类，不可跳过、不可输出“无法判断”。4. 仅输出类别名称，无其他描述性语言。输出格式（五选一）：请严格按照以下格式输出行为类别：高风险 / 中风险 / 低风险 / 无风险 / 非楼道",
                        "role": "user"
                    },
                    {
                        "content": label,
                        "role": "assistant"
                    }
                ],
                "images": [
                    os.path.join(image_folder_path, image_name)
                ]
            }

            data_list.append(item)

    # 写入 JSON 文件
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(data_list, f, ensure_ascii=False, indent=2)

    print(f"转换完成，已保存为：{output_json_path}")
