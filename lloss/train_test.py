from config import *
import torch
from tqdm import tqdm
import torchvision.transforms as T
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
##
# Loss Prediction Loss
def LossPredLoss(input, target, margin=1.0, reduction='mean'):
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape
    loss = 0
    
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


def test(models, dataloaders, mode='val'):
    assert mode == 'val' or mode == 'test'
    models['backbone'].eval()
    models['module'].eval()
    
    metrics = {}
    preds, labels = [], []
    
    with torch.no_grad():
        for data in dataloaders['test']:
            inputs = data[0].to(device)
            label = data[1].to(device)

            scores = models['backbone'](inputs)
            
            labels.extend(label.detach().tolist())
            preds.extend(scores.argmax(axis=1).detach().tolist())
            
    metrics['accuracy'] = accuracy_score(y_pred=preds, y_true=labels)
    metrics['f1_score'] = f1_score(y_pred=preds, y_true=labels, average='weighted')
    metrics['precision'] = precision_score(y_pred=preds, y_true=labels, average='weighted')
    metrics['recall'] = recall_score(y_pred=preds, y_true=labels, average='weighted')
            
    return metrics


def train_epoch(models, extractor, criterion, optimizers, dataloaders, epoch, epoch_loss):

    models['backbone'].train()
    models['module'].train()
    loss = 0
    for data in tqdm(dataloaders['train'], leave=False, total=len(dataloaders['train'])):
        inputs = data[0].to(device)
        labels = data[1].to(device)

        optimizers['backbone'].zero_grad()
        optimizers['module'].zero_grad()

        predicted = models['backbone'](inputs) 
        target_loss = criterion(predicted, labels)
        
        features_dic = extractor(inputs)
        features = [features_dic['layer1'],features_dic['layer2'],features_dic['layer3'],features_dic['layer4']]
        
        if epoch > epoch_loss:
            features[0] = features[0].detach()
            features[1] = features[1].detach()
            features[2] = features[2].detach()
            features[3] = features[3].detach()

        pred_loss = models['module'](features)
        pred_loss = pred_loss.view(pred_loss.size(0))
        m_module_loss   = LossPredLoss(pred_loss, target_loss, margin=MARGIN)
        m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)        
        loss            = m_backbone_loss + WEIGHT * m_module_loss 

        loss.backward()
        optimizers['backbone'].step()
        optimizers['module'].step()
        
        if epoch % 20 == 0 and epoch != 0 :
            print(f"epoch : {epoch} || loss : {loss}")


def train(models, extractor, criterion, optimizers, schedulers, dataloaders, num_epochs, epoch_loss):
    print('>> Train a Model.')

    for epoch in range(num_epochs):

        train_epoch(models, extractor, criterion, optimizers, dataloaders, epoch, epoch_loss)
        schedulers['backbone'].step()
        schedulers['module'].step()

    print('>> Finished.')