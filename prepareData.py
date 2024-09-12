import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image
import config

# 定义数据转换
transform = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    # transforms.RandomRotation(10),
    # transforms.RandomResizedCrop(224),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    # transforms.RandomGrayscale(p=0.1),
    # transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),  # 添加高斯模糊
    # transforms.RandomErasing(),
])


def get_datasets():
    # 加载数据集
    train_data = ImageFolder(root=config.train_dir, transform=transform)
    valid_data = ImageFolder(root=config.valid_dir, transform=transform)
    test_data = ImageFolder(root=config.test_dir, transform=transform)

    # 数据集大小
    train_data_size = len(train_data)
    valid_data_size = len(valid_data)
    test_data_size = len(test_data)
    print("The size of Train_data is {}".format(train_data_size))
    print("The size of Valid_data is {}".format(valid_data_size))
    print("The size of Test_data is {}".format(test_data_size))

    # dataloader进行数据集的加载
    train_dataloader = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=1, pin_memory=True)
    valid_dataloader = DataLoader(valid_data, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=1, pin_memory=True)
    test_dataloader = DataLoader(test_data, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=1, pin_memory=True)

    return train_dataloader, valid_dataloader, test_dataloader