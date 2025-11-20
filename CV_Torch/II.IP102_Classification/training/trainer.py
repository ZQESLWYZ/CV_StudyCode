import sys
import os
import torch
import torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import read_yaml
from models.resnet import ResNet34
from data.dataloader import get_dataloader
from losses import get_criterion
from optimizers import get_optimizer
from schedulers import get_scheduler

class trainer():
    """The class of model train"""
    def __init__(self, config, model_name='resnet34'):
        
        self.config = config
        self.device = config['train']['device']
        self.model_name = config['model']['name']
        
        self.epoch = config['train']['num_epochs']
        self.lr = config['train']['learning_rate']
        self.weight_decay = config['train']['weight_decay'] 
        self.early_stop = config['train']['early_stopping_patience']
        
        self.num_class = config['model']['num_classes']
        self.pretrained = config['model']['pretrained']
        
        self.exp_name = config['exp']['name']
        self.exp_dir_name = config['exp']['save_dir']
        self.log_interval = config['exp']['log_interval']
        
    def start_train(self):
        if self.model_name == 'resnet34':
            model = ResNet34(self.num_class).to(self.device)
        # TODO: ADD MORE MODELS
        else:
            pass
        
        train_loader, val_loader = get_dataloader(self.config)
        optimizer = get_optimizer(model, self.config)
        criterion = get_criterion(self.config)
        schedulers = get_scheduler(optimizer, self.config, train_loader)
            
        for epoch in range(self.epoch):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                optimizer.zero_grad()
                output = model(inputs)
                
                loss = criterion(output, targets)
                loss.backward()
                optimizer.step()
            print(loss)

if __name__ == "__main__":
    config = read_yaml(r"CV_Torch\II.IP102_Classification\configs\train_config.yaml")
    trainera = trainer(config)
    trainera.start_train()