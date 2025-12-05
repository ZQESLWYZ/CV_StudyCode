"""
# File       : train.py
# Author     : ZQESLWYZ
# version    : python 3.10
# Software   : VSCode
# Date       : 2025-12-05
"""
import os
import argparse
import time
import datetime

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from utils.data import get_data_loaders
from model.base_model import BaseModel
from utils.train_val_utils import train_one_epoch, evaluate

def main(args):
    # 选择训练设备:cuda/cpu
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # 创建网络权重保存的文件夹
    script_dir = os.path.dirname(os.path.abspath(__file__))
    weights_dir = os.path.join(script_dir, 'weights')
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    # 创建Tensorboard
    log_writer = SummaryWriter(os.path.join(script_dir, 'logs', f'{args.exp_name}'))
    
    # 获取数据集和加载器
    train_dataset, val_dataset, train_dataloader, val_dataloader = get_data_loaders(args.data_path, args.batch_size)

    # 加载模型
    model = BaseModel(args.model_name, args.num_classes).to(device)

    # 加载优化器
    optimizer = optim.AdamW(params=model.parameters(), lr=args.lr)
    
    # 初始化最佳准确率
    best_acc = 0.0
    
    # 计时并开始训练
    start = time.time()
    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(
            model=model,
            optimizer=optimizer,
            dataloader=train_dataloader,
            device=device,
            epoch=epoch
        )
        
        # val
        val_loss, val_acc = evaluate(
            model=model,
            dataloader=val_dataloader,
            device=device,
            epoch=epoch
        )        
        
        # 记录数据
        tags = ['train_loss', 'train_acc', 'val_loss', 'val_acc', 'learning_rate']
        log_writer.add_scalar(tags[0], train_loss, epoch)
        log_writer.add_scalar(tags[1], train_acc, epoch)
        log_writer.add_scalar(tags[2], val_loss, epoch)
        log_writer.add_scalar(tags[3], val_acc, epoch)
        log_writer.add_scalar(tags[4], optimizer.param_groups[0]['lr'], epoch)
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(weights_dir, f"{args.model_name}_best.pth"))
    torch.save(model.state_dict(), os.path.join(weights_dir, f"{args.model_name}_last.pth"))                
    end = time.time()
    print(f"训练完成！\n用时{(end-start):.2f}")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='alexnet')
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--data_path', type=str, default='Dataset/flower5')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--exp_name', type=str, default=time.time())
    opt = parser.parse_args()
    
    print(opt)
    main(opt)