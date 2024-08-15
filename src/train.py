import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from dataloader import SquatDataset
from torchvision.models import mobilenet_v3_small

#训练轮数
num_epochs = 100

# 定义模型
model = mobilenet_v3_small(pretrained=True)

# 修改模型的输出层 - 三个独立的输出头
# action_stage 输出头: 5 分类
model.classifier[3] = nn.Linear(1024, 5)

# error_type 输出头: 6 分类
model.error_classifier = nn.Sequential(
    nn.Linear(1024, 6),
)

# valid_squat 输出头:  二分类
model.valid_classifier = nn.Sequential(
    nn.Linear(1024, 1),
)

# 将模型移动到设备 (CPU 或 GPU)
device = torch.device('cuda')
model = model.to(device)

# 定义损失函数
criterion_action = nn.CrossEntropyLoss()   # 用于 action_stage (5 分类)
criterion_error = nn.CrossEntropyLoss()   # 用于 error_type (6 分类)
criterion_valid = nn.BCEWithLogitsLoss()  # 用于 valid_squat (二分类)

# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 数据加载
json_files = [
    '../data/1.json','../data/2.json','../data/3.json','../data/4.json','../data/5.json','../data/6.json',
    '../data/7.json','../data/8.json','../data/9.json','../data/10.json','../data/11.json','../data/12.json',
    '../data/13.json','../data/14.json','../data/15.json','../data/16.json'
]
keypoints_folder = '../data/keypoints'
video_folder = '../data/videos'

dataset = SquatDataset(json_files, keypoints_folder)
train_size = int(0.8 * len(dataset))  # 80% 用于训练
val_size = len(dataset) - train_size  # 20% 用于验证
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 定义保存路径
save_dir = './checkpoints'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 初始化最佳验证损失
best_val_loss = float('inf')

# 训练循环
for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    train_losses = []
    train_action_correct = 0
    train_error_correct = 0
    train_valid_correct = 0
    train_total = 0
    for i, (keypoints, action_stage, error_type, valid_squat) in enumerate(train_dataloader):
        # 将数据移动到设备
        keypoints = keypoints.to(device)
        action_stage = action_stage.to(device)
        error_type = error_type.to(device)
        valid_squat = valid_squat.to(device)

        # 清空梯度
        optimizer.zero_grad()

        # 前向传播
        action_stage_output = model(keypoints)
        error_type_output = model.error_classifier(model.features(keypoints))
        valid_squat_output = model.valid_classifier(model.features(keypoints))

        # 计算损失
        loss_action = criterion_action(action_stage_output, action_stage)
        loss_error = criterion_error(error_type_output, error_type)
        loss_valid = criterion_valid(valid_squat_output, valid_squat.unsqueeze(1).float())

        # 加权求和损失 (提高错误纠正和有效深蹲计数的权重)
        loss = loss_action + 2 * loss_error + 2 * loss_valid

        # 反向传播
        loss.backward()

        # 更新参数
        optimizer.step()

        # 记录训练指标
        train_losses.append(loss.item())
        _, predicted_action = torch.max(action_stage_output.data, 1)
        train_action_correct += torch.sum(torch.eq(predicted_action, action_stage)).item()
        _, predicted_error = torch.max(error_type_output.data, 1)
        train_error_correct += torch.sum(torch.eq(predicted_error, error_type)).item()
        predicted_valid = (torch.sigmoid(valid_squat_output) > 0.5).float()
        train_valid_correct += torch.sum(predicted_valid == valid_squat.unsqueeze(1).float()).item()
        train_total += action_stage.size(0)

    # 计算平均损失和准确率
    train_loss = np.mean(train_losses)
    train_action_accuracy = 100 * train_action_correct / train_total
    train_error_accuracy = 100 * train_error_correct / train_total
    train_valid_accuracy = 100 * train_valid_correct / train_total

    # 打印训练指标
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss:.4f}, '
          f'Action Accuracy: {train_action_accuracy:.2f}%, '
          f'Error Accuracy: {train_error_accuracy:.2f}%, '
          f'Valid Accuracy: {train_valid_accuracy:.2f}%')

    # 验证阶段
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 关闭梯度计算
        val_losses = []
        val_action_correct = 0
        val_error_correct = 0
        val_valid_correct = 0
        val_total = 0
        for i, (keypoints, action_stage, error_type, valid_squat) in enumerate(val_dataloader):
            # 将数据移动到设备
            keypoints = keypoints.to(device)
            action_stage = action_stage.to(device)
            error_type = error_type.to(device)
            valid_squat = valid_squat.to(device)

            # 前向传播
            action_stage_output = model(keypoints)
            error_type_output = model.error_classifier(model.features(keypoints))
            valid_squat_output = model.valid_classifier(model.features(keypoints))

            # 计算损失
            loss_action = criterion_action(action_stage_output, action_stage)
            loss_error = criterion_error(error_type_output, error_type)
            loss_valid = criterion_valid(valid_squat_output, valid_squat.unsqueeze(1).float())

            # 加权求和损失
            loss = loss_action + 2 * loss_error + 2 * loss_valid

            # 记录验证指标
            val_losses.append(loss.item())
            _, predicted_action = torch.max(action_stage_output.data, 1)
            val_action_correct += torch.sum(torch.eq(predicted_action, action_stage)).item()
            _, predicted_error = torch.max(error_type_output.data, 1)
            val_error_correct += torch.sum(torch.eq(predicted_error, error_type)).item()
            predicted_valid = (torch.sigmoid(valid_squat_output) > 0.5).float()
            val_valid_correct += torch.sum(predicted_valid == valid_squat.unsqueeze(1).float()).item()
            val_total += action_stage.size(0)

        # 计算平均损失和准确率
        val_loss = np.mean(val_losses)
        val_action_accuracy = 100 * val_action_correct / val_total
        val_error_accuracy = 100 * val_error_correct / val_total
        val_valid_accuracy = 100 * val_valid_correct / val_total

        # 打印验证指标
        print(f'Validation Loss: {val_loss:.4f}, '
              f'Validation Action Accuracy: {val_action_accuracy:.2f}%, '
              f'Validation Error Accuracy: {val_error_accuracy:.2f}%, '
              f'Validation Valid Accuracy: {val_valid_accuracy:.2f}%')

        # 模型保存
        # 保存 checkpoint
        checkpoint_path = os.path.join(save_dir, f'model_epoch_{epoch + 1}.pth')
        torch.save(model.state_dict(), checkpoint_path)

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(save_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)

        print(f'Checkpoint saved to {checkpoint_path}')
        print(f'Best model saved to {best_model_path}')