import os
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

if __name__ == '__main__':
    # 不使用随机组合，每个增强方式单独使用
    augmentations = [
        transforms.Compose([
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        ]),
        transforms.Compose([
            transforms.GaussianBlur(kernel_size=3),
            transforms.ColorJitter(hue=0.1),
        ]),
        transforms.Compose([
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        ]),
        transforms.Compose([
            transforms.RandomPerspective(distortion_scale=0.2, p=1.0),
        ]),
        transforms.Compose([
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]),
        transforms.Compose([
            transforms.Lambda(lambda img: transforms.functional.adjust_brightness(img, 0.2)),  # 模拟夜晚
        ]),
        transforms.Compose([
            transforms.Resize((512, 512)),  #  新增的第七种增强：强制拉伸
        ]),
    ]

    # 设置每个类别的增强倍数（原图+增强之后尽量达到2000张）
    class_aug_multipliers = {
        '高风险': 18,
        '中风险': 4,
        '低风险': 3,
        '非楼道': 11,
        '无风险': 0  # 不增强
    }

    # 输入输出路径
    input_dir = "/root/images/train"
    output_dir = "/root/autodl-tmp/data-aug"
    label_file = "/root/images/train.txt"
    output_label_file = "/root/autodl-tmp/train-aug.txt"

    os.makedirs(output_dir, exist_ok=True)

    # 读取原始标签
    label_dict = {}
    with open(label_file, 'r', encoding='utf-8') as f:
        for line in f:
            name, label = line.strip().split('\t')
            label_dict[name] = label

    # 对每类图像数量统计
    class_counter = {}

    with open(output_label_file, 'w', encoding='utf-8') as fout:
        for fname in tqdm(os.listdir(input_dir)):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            if fname not in label_dict:
                continue

            label = label_dict[fname]
            class_counter[label] = class_counter.get(label, 0) + 1

            img_path = os.path.join(input_dir, fname)
            img = Image.open(img_path).convert("RGB")
            base = os.path.splitext(fname)[0]

            # 保存原图
            orig_name = f"{base}_orig.jpg"
            img.save(os.path.join(output_dir, orig_name))
            fout.write(f"{orig_name}\t{label}\n")

            # 增强
            multiplier = class_aug_multipliers.get(label, 0)
            for i in range(multiplier):
                aug = augmentations[i % len(augmentations)]  # 每次按顺序选用增强方式
                aug_img = aug(img)
                aug_name = f"{base}_aug{i+1}.jpg"
                aug_img.save(os.path.join(output_dir, aug_name))
                fout.write(f"{aug_name}\t{label}\n")

    # 输出每类图像总数（原图+增强图）
    print("\n统计信息（原图+增强图）:")
    for cls, count in class_counter.items():
        multiplier = class_aug_multipliers.get(cls, 0)
        total = count * (1 + multiplier)
        print(f"{cls}：原图{count} × 增强{multiplier} = {total}张")
