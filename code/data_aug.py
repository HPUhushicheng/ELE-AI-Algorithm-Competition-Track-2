import os
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

if __name__ == '__main__':
    # 6种增强方式（不改变原图尺寸）
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
    ]

    # 输入输出路径
    input_dir = "/root/images/train"
    output_dir = "/root/autodl-tmp/project/LLaMA-Factory/data/train-aug"
    label_file = "/root/images/train.txt"
    output_label_file = "/root/autodl-tmp/project/LLaMA-Factory/data/train-aug.txt"

    os.makedirs(output_dir, exist_ok=True)

    # 读取原始标签
    label_dict = {}
    with open(label_file, 'r', encoding='utf-8') as f:
        for line in f:
            name, label = line.strip().split('\t')
            label_dict[name] = label

    with open(output_label_file, 'w', encoding='utf-8') as fout:
        for fname in tqdm(os.listdir(input_dir)):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            if fname not in label_dict:
                continue

            label = label_dict[fname]
            img_path = os.path.join(input_dir, fname)
            img = Image.open(img_path).convert("RGB")
            base = os.path.splitext(fname)[0]

            # 保存原图
            orig_name = f"{base}_orig.jpg"
            img.save(os.path.join(output_dir, orig_name))
            fout.write(f"{orig_name}\t{label}\n")

            # 增强后保存
            for i, aug in enumerate(augmentations):
                aug_img = aug(img)
                aug_name = f"{base}_aug1{i+1}.jpg"
                aug_img.save(os.path.join(output_dir, aug_name))
                fout.write(f"{aug_name}\t{label}\n")
