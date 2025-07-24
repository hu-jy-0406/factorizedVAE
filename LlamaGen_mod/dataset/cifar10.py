import os
from torchvision.datasets import ImageFolder
from LlamaGen.dataset.custom_dataset import CustomDataset

def build_cifar10(args, transform):
    return ImageFolder(args.data_path, transform=transform)

def build_cifar10_code(type, args):
    if type == 'train': 
        feature_dir = f"{args.train_code_path}/cifar10{args.image_size}_codes"
        label_dir = f"{args.train_code_path}/cifar10{args.image_size}_labels"
    elif type == 'val':
        feature_dir = f"{args.val_code_path}/cifar10{args.image_size}_codes"
        label_dir = f"{args.val_code_path}/cifar10{args.image_size}_labels"
    else:
        raise ValueError(f"Unknown type: {type}, should be 'train' or 'val'")
    assert os.path.exists(feature_dir) and os.path.exists(label_dir), \
        f"please first run: bash scripts/autoregressive/extract_codes_c2i.sh ..."
    return CustomDataset(feature_dir, label_dir)
    