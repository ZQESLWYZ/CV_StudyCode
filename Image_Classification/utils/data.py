"""
# File       : utils/data.py
# Author     : ZQESLWYZ
# version    : python 3.10
# Software   : VSCode
# Date       : 2025-12-05
"""

import os
import json
import random
from typing import Tuple, List, Optional, Dict
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def read_split_data(
    root: str, 
    val_rate: float = 0.2, 
    plot_image: bool = True
) -> Tuple[List[str], List[int], List[str], List[int]]:
    """
    读取数据集并按比例划分为训练集和验证集
    
    Args:
        root: 数据集根目录路径，每个子目录代表一个类别
        val_rate: 验证集比例，默认0.2（20%）
        plot_image: 是否显示类别分布图，默认False
        
    Returns:
        train_paths: 训练集图片路径列表
        train_labels: 训练集标签列表
        val_paths: 验证集图片路径列表
        val_labels: 验证集标签列表
        
    Raises:
        AssertionError: 当数据集根目录不存在时抛出
    """
    random.seed(0)  # 设置随机种子保证结果可复现
    
    # 验证数据集根目录是否存在
    assert os.path.exists(root), f"数据集目录 {root} 不存在"
    
    # 获取所有类别（子目录名称）
    flower_classes = [
        cla for cla in os.listdir(root) 
        if os.path.isdir(os.path.join(root, cla))
    ]
    flower_classes.sort()  # 排序保证一致性
    
    # 创建类别到索引的映射字典
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(flower_classes)}
    
    # 保存类别映射到JSON文件
    script_last_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(script_last_dir, 'class_indices.json'), 'w') as f:
        json.dump({idx: cls_name for cls_name, idx in class_to_idx.items()}, f, indent=4)
    
    # 初始化数据容器
    train_paths, train_labels = [], []
    val_paths, val_labels = [], []
    class_counts = []  # 各类别样本数量
    
    # 支持的图片格式
    valid_extensions = ('.jpg', '.JPG', '.png', '.PNG', '.jpeg', '.JPEG')
    
    # 遍历每个类别文件夹
    for class_name in flower_classes:
        class_path = os.path.join(root, class_name)
        class_idx = class_to_idx[class_name]
        
        # 获取当前类别所有图片路径
        images = [
            os.path.join(class_path, img) for img in os.listdir(class_path)
            if img.lower().endswith(valid_extensions)
        ]
        
        # 随机采样验证集
        val_size = int(len(images) * val_rate)
        val_samples = random.sample(images, val_size)
        
        # 划分训练集和验证集
        for img_path in images:
            if img_path in val_samples:
                val_paths.append(img_path)
                val_labels.append(class_idx)
            else:
                train_paths.append(img_path)
                train_labels.append(class_idx)
        
        class_counts.append(len(images))  # 记录当前类别样本数
    
    # 输出数据集统计信息
    print(f"Flower 数据集统计:")
    print(f" 总样本数: {sum(class_counts)}")
    print(f" 训练集: {len(train_paths)} 张图片")
    print(f" 验证集: {len(val_paths)} 张图片")
    print(f" 类别数: {len(flower_classes)}")
    
    # 绘制类别分布图
    if plot_image:
        _plot_class_distribution(flower_classes, class_counts)
    
    return train_paths, train_labels, val_paths, val_labels


def _plot_class_distribution(class_names: List[str], class_counts: List[int]) -> None:
    """绘制类别分布柱状图"""
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(class_names)), class_counts, align='center', alpha=0.7)
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
    
    # 在每个柱子上方添加数量标签
    for i, count in enumerate(class_counts):
        plt.text(i, count + 0.5, str(count), ha='center', va='bottom')
    
    plt.xlabel('Class')
    plt.ylabel('Sample_Num')
    plt.title('Dataset Classes')
    plt.tight_layout()
    plt.show()


class MyDataset(Dataset):
    """自定义数据集类，继承自torch.utils.data.Dataset"""
    
    def __init__(
        self, 
        image_paths: List[str], 
        image_labels: List[int], 
        transform: Optional[transforms.Compose] = None
    ):
        """
        初始化数据集
        
        Args:
            image_paths: 图片路径列表
            image_labels: 对应的标签列表
            transform: 数据增强/预处理变换
        """
        self.image_paths = image_paths
        self.image_labels = image_labels
        self.transform = transform
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        获取单个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            image: 预处理后的图片张量
            label: 对应的标签
            
        Raises:
            ValueError: 当图片不是RGB模式时抛出
        """
        # 加载图片
        img_path = self.image_paths[idx]
        img = Image.open(img_path)
        
        # 验证图片模式
        if img.mode != 'RGB':
            raise ValueError(f"图片 {img_path} 不是RGB模式，当前模式: {img.mode}")
        
        # 获取标签
        label = self.image_labels[idx]
        
        # 应用数据增强/预处理
        if self.transform:
            img = self.transform(img)
        
        return img, label
    
    @staticmethod
    def collate_fn(batch: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        批处理函数，将多个样本组合成一个批次
        
        Args:
            batch: 样本列表，每个元素为(image, label)元组
            
        Returns:
            batch_images: 批处理后的图片张量 [batch_size, channels, height, width]
            batch_labels: 批处理后的标签张量 [batch_size]
        """
        images, labels = zip(*batch)  # 解压批次
        images = torch.stack(images, dim=0)  # 堆叠图片
        labels = torch.tensor(labels)  # 转换为张量
        return images, labels


def get_data_loaders(
    data_root: str, 
    batch_size: int = 32
) -> Tuple[Dataset, Dataset, DataLoader, DataLoader]:
    """
    获取训练集和验证集的数据加载器
    
    Args:
        data_root: 数据集根目录
        batch_size: 批次大小，默认32
        
    Returns:
        train_dataset: 训练集Dataset对象
        val_dataset: 验证集Dataset对象
        train_loader: 训练集DataLoader
        val_loader: 验证集DataLoader
    """
    # 划分数据集
    train_paths, train_labels, val_paths, val_labels = read_split_data(data_root)
    
    # 定义数据预处理流水线
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),  # 随机裁剪并缩放
            transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
            transforms.ToTensor(),  # 转换为张量
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet标准化参数
                               std=[0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),  # 调整大小
            transforms.CenterCrop(224),  # 中心裁剪
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    }
    
    # 创建数据集
    train_dataset = MyDataset(train_paths, train_labels, data_transforms['train'])
    val_dataset = MyDataset(val_paths, val_labels, data_transforms['val'])
    
    # 计算合适的工作进程数
    num_workers = 0
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # 训练时打乱数据
        num_workers=num_workers,
        pin_memory=True,  # 启用内存锁页，加速GPU传输
        collate_fn=MyDataset.collate_fn,
        drop_last=True  # 丢弃最后一个不完整的批次
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # 验证时不打乱
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=MyDataset.collate_fn,
        drop_last=False
    )
    
    return train_dataset, val_dataset, train_loader, val_loader

