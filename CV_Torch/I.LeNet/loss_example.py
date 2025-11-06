from model import LeNet
import torch
import torch.nn as nn
import torch.nn.functional as F

if __name__ == "__main__":

    net = LeNet(10)
    input = torch.rand(10, 3, 32, 32)
    target = torch.rand(10, 10)
    output = net(input)
    
    # 创建一个MSE loss的实例
    criterion = nn.MSELoss()
    
    # args: output, target
    loss = criterion(output, target)   
    print(loss)
    
    net.zero_grad()
    # 反向传播前的梯度信息
    print(net.conv1.bias.grad)
    print(net.conv1.bias)
    
    loss.backward()
    
    # 反向传播后的梯度信息,可知，只有进行optimizer.step()才更新梯度
    print(net.conv1.bias.grad)
    print(net.conv1.bias)