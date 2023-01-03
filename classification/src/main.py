from config import *

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
import numpy as np
import torch.backends.cudnn as cudnn
import random

import timm
from data import train_data, test_data
from train import *

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(0)

print(f" train_data length : {len(train_data)}")
print(f" test_data length : {len(test_data)}")

train_loader = DataLoader(train_data, batch_size = BATCH)
test_loader = DataLoader(test_data, batch_size = 1)
dataloaders  = {'train': train_loader, 'test': test_loader}

model = timm.create_model('resnet50', pretrained = True, num_classes = 2).to(device)
criterion = nn.CrossEntropyLoss()
optim = optim.Adam(model.parameters(), lr = LR)

scheduler = lr_scheduler.MultiStepLR(optim, milestones = [30,60,90])

train(model, criterion, optim, scheduler, dataloaders, EPOCH)
metrics = test(model, dataloaders)

print(f"accuracy : {metrics['accuracy']:.4f}")
print(f"f1 score : {metrics['f1_score']:.4f}")
print(f"precision : {metrics['precision']:.4f}")
print(f"recall : {metrics['recall']:.4f}")