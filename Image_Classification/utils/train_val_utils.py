"""
# File       : utils/train_val_utils.py
# Author     : ZQESLWYZ
# version    : python 3.10
# Software   : VSCode
# Date       : 2025-12-05
"""
import sys

import torch
import torch.nn as nn
from tqdm import tqdm

def train_one_epoch(model: nn.Module, 
                    optimizer: torch.optim.Optimizer,
                    dataloader: torch.utils.data.DataLoader, 
                    device: torch.device, 
                    epoch: int):
    # 设置模型为训练模式且清零梯度
    model.train()
    optimizer.zero_grad()
    
    # 初始化损失函数：分类用交叉熵
    loss_fun = nn.CrossEntropyLoss().to(device)  # 将损失函数也移到设备
    
    # 初始化累计量，用于进度条显示
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)  # 累计正确预测的样本量
    sample_num = 0  # 累计样本数
    
    # 使用tqdm包裹dataloader
    dataloader = tqdm(dataloader, desc=f'Training Epoch {epoch}')
    
    # 对每一批次进行循环:
    for step, (imgs, labels) in enumerate(dataloader, start=1):
        # 获取批次大小（修正：应该是imgs.size(0)而不是imgs[0]）
        batch_size = imgs.size(0)
        sample_num += batch_size
        
        # 将数据移动到设备
        imgs = imgs.to(device)
        labels = labels.to(device)  # 提前移动labels到设备
        
        # 前向传播
        pred = model(imgs)
        
        # 计算预测类别
        pred_labels = torch.max(pred, dim=1)[1]
        
        # 计算正确预测样本数（修正：应该是累加而不是赋值）
        correct = torch.eq(labels, pred_labels).sum()
        accu_num += correct  # 累加正确预测数
        
        # 计算损失
        loss = loss_fun(pred, labels)  # 现在labels已经在设备上了
        
        # 计算累计损失
        accu_loss += loss.detach()
        
        # 计算平均损失和准确率
        avg_loss = accu_loss.item() / step
        avg_acc = accu_num.item() / sample_num
        
        # 更新进度条描述
        dataloader.set_description(f"[Train Epoch {epoch}] Loss: {avg_loss:.3f}, Acc: {avg_acc:.3f}")
        
        # 反向传播更新梯度
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return avg_loss, avg_acc
    
@torch.no_grad()
def evaluate(model: nn.Module, dataloader: torch.utils.data.DataLoader, 
             device: torch.device, epoch: int):
    
    # 设为评估模式不计算梯度
    model.eval()

    # 初始化损失函数：分类用交叉熵
    loss_fun = nn.CrossEntropyLoss().to(device)  # 将损失函数也移到设备
    
    # 初始化累计量，用于进度条显示
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)  # 累计正确预测的样本量
    sample_num = 0  # 累计样本数
    
    dataloader = tqdm(dataloader, desc=f'Evaluating Epoch {epoch}')

    # 对每一批次进行循环:
    for step, (imgs, labels) in enumerate(dataloader, start=1):
        # 获取批次大小
        batch_size = imgs.size(0)
        sample_num += batch_size
        
        # 将数据移动到设备
        imgs = imgs.to(device)
        labels = labels.to(device)  # 修正：labels也需要移动到设备
        
        # 前向传播
        pred = model(imgs)
        
        # 计算预测类别
        pred_labels = torch.max(pred, dim=1)[1]
        
        # 计算正确预测样本数
        correct = torch.eq(labels, pred_labels).sum()
        accu_num += correct  # 累加正确预测数
        
        # 计算损失
        loss = loss_fun(pred, labels)  # 修正：labels已经在设备上
        
        # 计算累计损失
        accu_loss += loss.detach()
        
        # 计算平均损失和准确率
        avg_loss = accu_loss.item() / step
        avg_acc = accu_num.item() / sample_num
        
        # 修正：验证的描述应该是Eval
        dataloader.set_description(f"[Eval Epoch {epoch}] Loss: {avg_loss:.3f}, Acc: {avg_acc:.3f}")
        
    return avg_loss, avg_acc