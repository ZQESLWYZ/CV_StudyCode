from model import LeNet
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

if __name__ == "__main__":
    net = LeNet(10)
    # 实例化优化器类
    optimizer = optim.Adam(net.parameters(), lr=1e-2)
    
    # 
    input = torch.rand(10, 3, 32, 32)
    target = torch.rand(10, 10)
    output = net(input)
    
    # 创建一个MSE loss的实例
    criterion = nn.MSELoss()
    
    # args: output, target
    loss = criterion(output, target)   
    print(loss)
    
    optimizer.zero_grad()
    
    print(net.conv1.bias.grad)
    print(net.conv1.bias)
    
    loss.backward()
    optimizer.step()
    
    print(net.conv1.bias.grad)
    print(net.conv1.bias)
    