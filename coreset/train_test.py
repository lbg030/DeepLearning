from config import *
import torch
from tqdm import tqdm
import numpy as np
import torchvision.transforms as T
import models.resnet as resnet
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import warnings
import torch.optim as optim

warnings.filterwarnings('ignore')
##
# Loss Prediction Loss
def LossPredLoss(input, target, margin=1.0, reduction='mean'):
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape
    
    input = (input - input.flip(0))[:len(input)//2] # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
    target = (target - target.flip(0))[:len(target)//2]
    target = target.detach()

    one = 2 * torch.sign(torch.clamp(target, min=0)) - 1 # 1 operation which is defined by the authors
    
    if reduction == 'mean':
        loss = torch.sum(torch.clamp(margin - one * input, min=0))
        loss = loss / input.size(0) # Note that the size of input is already halved
    elif reduction == 'none':
        loss = torch.clamp(margin - one * input, min=0)
    else:
        NotImplementedError()
    
    return loss


def test(models, epoch, method, dataloaders, mode='val'):
    assert mode == 'val' or mode == 'test'
    metrics = {}
    results = {}
    preds, labels, probs,features = [], [], [], []
    models['backbone'].eval()

    with torch.no_grad():
        for data in dataloaders[mode]:
            inputs = data[0].to(device)
            label = data[1].to(device)
                
            output = models['backbone'](inputs)
            # pred = output.argmax(dim=1)
            # pred = torch.max(output.data, 1)
            
            # labels.append(label.item())
            # preds.append(pred.item())
            
            labels.extend(label.detach().tolist())
            preds.extend(output.argmax(axis=1).detach().tolist())
            # probs.append(output.softmax(dim=1).detach().cpu().numpy().ravel())

    metrics['accuracy'] = accuracy_score(y_pred=preds, y_true=labels)
    metrics['f1_score'] = f1_score(y_pred=preds, y_true=labels, average='weighted')
    metrics['precision'] = precision_score(y_pred=preds, y_true=labels, average='weighted')
    metrics['recall'] = recall_score(y_pred=preds, y_true=labels, average='weighted')

    # results['labels'] = labels
    # results['predictions'] = preds 
    # results['class_probability'] = class_probs

    return metrics, results

def train_epoch(models, method, criterion, optimizers, dataloaders, epoch, epoch_loss):
    models['backbone'].train()
    epoch_loss = 0
    
    global iters
    for data in tqdm(dataloaders['train'], leave=False, total=len(dataloaders['train'])):
        # with torch.cuda.device(device):
        inputs = data[0].to(device)
        labels = data[1].to(device)

        optimizers['backbone'].zero_grad()

        output = models['backbone'](inputs) 
        target_loss = criterion(output, labels)

        epoch_loss += target_loss.item()
        
        target_loss.backward()
        optimizers['backbone'].step()
    
    
    # df = pd.DataFrame(labels, columns = ['labels'])
    
    # df.to_csv("fst.csv", index = False) 
    epoch_loss /= len(dataloaders['train'].dataset)
    return epoch_loss

def train(models, method, criterion, optimizers, schedulers, dataloaders, num_epochs, epoch_loss):
    print('>> Train a Model.')
   
    best_acc = 0.
    
    for epoch in tqdm(range(num_epochs)):
        if epoch <= 160 :
            optim_backbone = optim.SGD(models['backbone'].parameters(), lr=0.1, 
                momentum=MOMENTUM, weight_decay=WDECAY)
            optimizers['backbone'] = optim_backbone
        else :
            optim_backbone = optim.SGD(models['backbone'].parameters(), lr=0.01, 
                momentum=MOMENTUM, weight_decay=WDECAY)
            optimizers['backbone'] = optim_backbone
        best_loss = torch.tensor([0.5]).to(device)
        unused_loss = train_epoch(models, method, criterion, optimizers, dataloaders, epoch, epoch_loss)

        schedulers['backbone'].step()

    print('>> Finished.')
    return unused_loss