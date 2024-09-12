import torch
import torchvision
from torchvision import datasets, transforms
import os, shutil
import matplotlib.pyplot as plt
import matplotlib
import pickle
import json

print(torch.__version__, torchvision.__version__)
# 数据转换
transform = transforms.Compose(
    [transforms.CenterCrop(224),  # 图片大小为224*224
     transforms.ToTensor(),  # 转换为0-1的张量
     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 标准化 -1到1
     ]
)

file_path = 'dataset/'
Image_dataset = {
    x: datasets.ImageFolder(
        root=os.path.join(file_path, x),
        transform=transform
    )
    for x in ['train', 'valid']
}
Image_data_loader = {
    x: torch.utils.data.DataLoader(
        dataset=Image_dataset[x],
        batch_size=64,  # 64张图一个批次读取
        shuffle=True
    )
    for x in ['train', 'valid']
}

# 查看标签
classes = Image_dataset['train'].classes
classes_index = Image_dataset['train'].class_to_idx
X_train, Y_train = next(iter(Image_data_loader['train']))
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]
img = torchvision.utils.make_grid(X_train)
img = img.numpy().transpose((1, 2, 0))
img = img * std + mean
plt.imshow(img)
plt.savefig('myplot.png')
train_dataloader = Image_data_loader['train']
val_dataloader = Image_data_loader['valid']
classes_to_index = classes_index

# 保存数据
with open('appRefer/train_dl.pkl', 'wb') as f:
    pickle.dump(train_dataloader, f)
with open('appRefer/val_dl.pkl', 'wb') as f:
    pickle.dump(val_dataloader, f)
with open('appRefer/index_to_class.json', mode='w', encoding='utf-8') as f:
    json.dump({v: k for k, v in classes_to_index.items()}, f, indent=4)
