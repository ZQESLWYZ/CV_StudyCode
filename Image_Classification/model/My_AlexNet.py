import torch 
import torch.nn as nn
import torch.functional as F

from torchinfo import summary
from torchvision.models import alexnet

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            
            nn.Conv2d(64, 192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),   
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),          
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(6, 6))
        
        self.classifier = nn.Sequential(
            
            # 输入 → Dropout(0.5) → Linear1 → ReLU → Dropout(0.5) → Linear2 → ReLU → Linear3 → ReLU → 输出
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes) 
        )
        
    def forward(self, x):
        return self.classifier(torch.flatten(self.avgpool(self.features(x)), 1))
    
if __name__ == "__main__":
    a_alexnet = AlexNet(10)
    summary(a_alexnet, (1, 3, 256, 256))