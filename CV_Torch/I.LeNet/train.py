import torch
import numpy as np
import torchvision
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from model import LeNet
from torch.utils.tensorboard import SummaryWriter

def main():
    # 定义超参数
    exp_name = "LeNet_CIFAR10"
    batch_size = 2048
    lr = 1e-4
    epochs = 100
    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    
    trainloader, testloader = get_dataloader(batch_size)
    train(lr, epochs, trainloader, testloader, classes, exp_name)

    # dataiter = iter(trainloader)
    # images, labels = next(dataiter)
    
    # imgshow(torchvision.utils.make_grid(images, nrow = 8))
    # print(f"{[classes[idx] for idx in labels]}")
    
    
def train(lr, epochs, trainloader, testloader, classes, exp_name) -> None:
    
    writer = SummaryWriter(f'CV_Torch/I.LeNet/log/{exp_name}')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    net = LeNet(10).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), lr)
    
    # 每个epoch
    for epoch in range(epochs):
        running_loss = 0.0
        # 每个batch
        for i, data in enumerate(trainloader):
            
            imgs, labels = data[0].to(device), data[1].to(device)
            
            output = net(imgs)
            
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss
        writer.add_scalar('Loss/train', running_loss, epoch)    
        print(f"batch_{i}: imgs_shape is {imgs.shape} labels_shape is {labels.shape}")
        print(f'[epoch: {epoch + 1}] loss: {running_loss:.3f}')
        running_loss = 0.0
    
    writer.close()
    
def imgshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
def get_dataloader(batch_size):
    # 数据集处理
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    batch_size = batch_size
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size, 
                                              shuffle=True, num_workers=0)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)
    
    testloader = torch.utils.data.DataLoader(testset, batch_size, 
                                              shuffle=True, num_workers=0)    
    return trainloader, testloader
    
    
if __name__ == "__main__":
    main()