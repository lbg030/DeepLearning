import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from config import *

def train(model, criterion, optimizer, scedulers, dataloader, epochs):
    print('>> Train a Model.')

    for epoch in range(epochs):

        train_epoch(model, criterion, optimizer, dataloader, epoch)
        scedulers.step()

    print('>> Finished.')
    
def train_epoch(model, criterion, optimizer, dataloader, epoch):
    model.train()
    
    for data in tqdm(dataloader['train']):
        input = data[0].to(device)
        label = data[1].to(device)
        
        optimizer.zero_grad()
        
        predicted = model(input)
        loss = criterion(predicted, label)
        
        loss.backward()
        
        optimizer.step()
        
        if epoch % 40 == 0 and epoch != 0:
            print(f"epoch : {epoch} || loss : {loss}")
            

def test(model, dataloader):
    model.eval()
    
    metrics = {}
    preds, labels = [], []
    
    with torch.no_grad():
        for data in dataloader['test']:
            input = data[0].to(device)
            label = data[1].to(device)
            predicted = model(input)
            
            labels.extend(label.detach().tolist())
            preds.extend(predicted.argmax(axis=1).detach().tolist())
            
    metrics['accuracy'] = accuracy_score(y_pred=preds, y_true=labels)
    metrics['f1_score'] = f1_score(y_pred=preds, y_true=labels, average='weighted')
    metrics['precision'] = precision_score(y_pred=preds, y_true=labels, average='weighted')
    metrics['recall'] = recall_score(y_pred=preds, y_true=labels, average='weighted')
            
    return metrics
