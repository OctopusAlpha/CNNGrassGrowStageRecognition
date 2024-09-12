import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import random
import matplotlib.pyplot as plt
import config
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = f"{config.save_model_dir}/model_epoch_30.pth"  # 替换为你保存的模型路径
grass_stage = config.grass_stage
test_dir = 'dataset/test/'  # 测试集路径


# 定义函数，根据config.model选择模型
def get_resnet_model(model_name, num_classes):
    if model_name == "resnet18":
        model = models.resnet18(pretrained=True)
    elif model_name == "resnet34":
        model = models.resnet34(pretrained=True)
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=True)
    elif model_name == "resnet101":
        model = models.resnet101(pretrained=True)
    elif model_name == "resnet152":
        model = models.resnet152(pretrained=True)
    else:
        raise ValueError(f"Unknown model name {model_name}")

    # 修改最后的全连接层
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    # 添加dropout层
    dropout = nn.Dropout(p=0.1)
    model = nn.Sequential(model, dropout)

    return model


# 图像预处理函数
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = transform(image).unsqueeze(0)  # 增加batch维度
    return image


# 随机选择10个样本并展示验证结果
def random_sample_validation(model_path):
    # 加载模型
    model = get_resnet_model(config.model, num_classes=config.NUM_CLASSES)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    # 收集所有图片路径和对应的真实标签
    image_paths = []
    labels = []
    for idx, flower in enumerate(grass_stage):
        folder_path = os.path.join(test_dir, flower)
        if not os.path.exists(folder_path):
            print(f"Folder {folder_path} does not exist!")
            continue

        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            image_paths.append(image_path)
            labels.append(idx)

    # 随机选择10个样本
    sampled_indices = random.sample(range(len(image_paths)), 10)
    sampled_images = [image_paths[i] for i in sampled_indices]
    sampled_labels = [labels[i] for i in sampled_indices]

    # 创建子图用于展示结果
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    for i, (image_path, label) in enumerate(zip(sampled_images, sampled_labels)):
        # 预处理图像
        image = preprocess_image(image_path)
        image = image.to(device)

        # 进行预测
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)

        # 转换图像格式用于展示
        image = image.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        image = std * image + mean
        image = np.clip(image, 0, 1)

        # 获取当前子图
        ax = axes[i // 5, i % 5]
        ax.imshow(image)
        ax.axis('off')
        ax.set_title(f"True: {grass_stage[label]}\nPred: {grass_stage[predicted.item()]}")

    plt.tight_layout()
    plt.savefig('random_samples.png')  # 保存图片到当前目录
    print("The image has been saved as 'random_samples.png'.")


# 测试整个测试集并计算准确率
def validate_model_on_testset(model_path):
    # 加载模型
    model = get_resnet_model(config.model, num_classes=config.NUM_CLASSES)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    correct = 0
    total = 0
    class_correct = [0] * len(grass_stage)
    class_total = [0] * len(grass_stage)

    # 遍历测试集中的每个类别文件夹
    for idx, flower in enumerate(grass_stage):
        folder_path = os.path.join(test_dir, flower)
        if not os.path.exists(folder_path):
            print(f"Folder {folder_path} does not exist!")
            continue

        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            image = preprocess_image(image_path)
            image = image.to(device)

            # 进行预测
            with torch.no_grad():
                outputs = model(image)
                _, predicted = torch.max(outputs, 1)

            # 统计正确预测的样本
            total += 1
            class_total[idx] += 1
            if predicted.item() == idx:
                correct += 1
                class_correct[idx] += 1

    # 打印总体准确率
    overall_accuracy = correct / total * 100
    print(f"Overall accuracy on the test set: {overall_accuracy:.2f}%")

    # 打印每类的准确率
    for i, flower in enumerate(grass_stage):
        accuracy = class_correct[i] / class_total[i] * 100 if class_total[i] > 0 else 0
        print(f"Accuracy for class {flower}: {accuracy:.2f}%")


# 测试模型在整个测试集上的表现
validate_model_on_testset(model_path)

# 随机选择并展示10个验证结果
random_sample_validation(model_path)
