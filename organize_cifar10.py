import os
from torchvision import datasets, transforms
from PIL import Image

def save_images(dataset, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    # 定义 CIFAR-10 默认类别（可根据需要修改）
    classes = dataset.classes
    # 为每个类别创建文件夹
    for cls in classes:
        class_dir = os.path.join(output_dir, cls)
        os.makedirs(class_dir, exist_ok=True)
    to_pil = transforms.ToPILImage()
    # 保存图片文件
    for idx in range(len(dataset)):
        img, label = dataset[idx]
        class_name = classes[label]
        class_dir = os.path.join(output_dir, class_name)
        # 使用编号作为文件名，可根据需要更改命名方式
        img_filename = os.path.join(class_dir, f"{idx + 1}.png")
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(img_filename)

def load_cifar():
    
    train = datasets.CIFAR10(root="/home/hjy22/repos/vqvae/datasets/cifar-10-python", train=True, download=False)

    val = datasets.CIFAR10(root="/home/hjy22/repos/vqvae/datasets/cifar-10-python", train=False, download=False)
    return train, val

def main():
    # 定义转换（这里直接转换为 PIL Image，不需要 tensor）
    train_dataset, val_dataset = load_cifar()
    save_images(train_dataset, "CIFAR10/train")
    save_images(val_dataset, "CIFAR10/val")

if __name__ == "__main__":
    main()