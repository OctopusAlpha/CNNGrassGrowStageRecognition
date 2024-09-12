import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
# from torch.utils.tensorboard import SummaryWriter
import config
from prepareData import get_datasets

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 创建一个函数，根据config.model选择模型
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


if __name__ == '__main__':
    # 获取指定的模型
    model = get_resnet_model(config.model, num_classes=config.NUM_CLASSES)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)


    # 定义Early Stopping类
    class EarlyStopping:
        def __init__(self, patience=5, min_delta=0):
            self.patience = patience
            self.min_delta = min_delta
            self.counter = 0
            self.best_score = None
            self.early_stop = False

        def __call__(self, val_loss):
            if self.best_score is None:
                self.best_score = val_loss
            elif val_loss > self.best_score - self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = val_loss
                self.counter = 0


    early_stopping = EarlyStopping(patience=15, min_delta=0.01)

    train_dataloader, valid_dataloader, _ = get_datasets()

    best_valid_accuracy = 0

    # 保存学习率
    learning_rates = []

    for epoch in range(config.EPOCHS):
        model.train()
        total_train_loss = 0
        correct_train = 0
        total_train = 0
        for data in train_dataloader:
            imgs, targets = data
            imgs, targets = imgs.to(device), targets.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += targets.size(0)
            correct_train += (predicted == targets).sum().item()

        train_accuracy = correct_train / total_train

        # 验证集
        model.eval()
        total_valid_loss = 0
        correct_valid = 0
        total_valid = 0
        with torch.no_grad():
            for data in valid_dataloader:
                imgs, targets = data
                imgs, targets = imgs.to(device), targets.to(device)

                outputs = model(imgs)
                loss = criterion(outputs, targets)
                total_valid_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_valid += targets.size(0)
                correct_valid += (predicted == targets).sum().item()

        valid_accuracy = correct_valid / total_valid
        valid_loss = total_valid_loss / len(valid_dataloader)

        # 动态调整学习率
        scheduler.step(valid_loss)

        # 保存当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)

        # 检查Early Stopping
        if early_stopping(valid_loss):
            print("Early stopping triggered")
            break

        print("Epoch: {}/{}, train loss: {:.5f}, train accuracy: {:.5f}, valid loss: {:.5f}, valid accuracy: {:.5f}, lr: {:.5f}".format(
            epoch + 1, config.EPOCHS, total_train_loss / len(train_dataloader), train_accuracy, valid_loss, valid_accuracy, current_lr))

        # 保存最高准确率模型
        if valid_accuracy > best_valid_accuracy:
            best_valid_accuracy = valid_accuracy
            torch.save(model.state_dict(), f"{config.save_model_dir}/best_model.pth")
            print(f"最高准确率模型已保存为 best_model.pth 文件")

        # 保存每十轮的模型
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"{config.save_model_dir}/model_epoch_{epoch + 1}.pth")
            print(f"模型已保存为 model_epoch_{epoch + 1}.pth 文件")

    # 最后保存模型
    torch.save(model.state_dict(), f"{config.save_model_dir}/model_final.pth")
    print("最终模型已保存为 model_final.pth 文件")

    # 保存学习率到文件
    with open(f"{config.save_model_dir}/learning_rates.txt", 'w') as f:
        for lr in learning_rates:
            f.write(f"{lr}\n")
    print("学习率已保存为 learning_rates.txt 文件")
