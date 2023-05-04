import random
import pandas as pd
import numpy as np

import re
import glob
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from tqdm.auto import tqdm
import timm
from data.dataset import train_loader, val_loader, test_loader
from config import *

import warnings
warnings.filterwarnings(action='ignore')

def main():
    model = timm.create_model("resnet50", pretrained = True, num_classes = 19).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr = LR)
    scheduler = CosineAnnealingLR(optimizer, T_max = 32, eta_min = 1e-5)

    infer_model = train(model, optimizer, train_loader, val_loader, scheduler)
    preds = inference(infer_model, test_loader, DEVICE)

    return preds

def validation(model, criterion, val_loader, device):
    model.eval()
    val_loss = []
    preds, true_labels = [], []

    with torch.no_grad():
        for imgs, labels in tqdm(val_loader):
            imgs = imgs.float().to(device)
            labels = labels.to(device)
            
            pred = model(imgs)
            
            loss = criterion(pred, labels)
            
            preds += pred.argmax(1).detach().cpu().numpy().tolist()
            true_labels += labels.detach().cpu().numpy().tolist()
            
            val_loss.append(loss.item())
        
        _val_loss = np.mean(val_loss)
        _val_score = f1_score(true_labels, preds, average='weighted')
    
    return _val_loss, _val_score


def train(model, optimizer, train_loader, val_loader, scheduler):
    best_score = 0
    best_model = None
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    scaler = torch.cuda.amp.GradScaler()
    model.train()

    for epoch in range(EPOCH):
        train_loss = []
        with tqdm(train_loader, unit='batch') as tepoch:
            for imgs, labels in tepoch:
                batch += 1
                optimizer.zero_grad()
                tepoch.set_description(f"Epoch {epoch}")
                imgs = imgs.float().to(DEVICE)
                labels = labels.to(DEVICE)          
                
                with torch.cuda.amp.autocast():

                    output = model(imgs).to(DEVICE)
                    loss = criterion(output, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss.append(loss.item())

                tepoch.set_postfix(
                    phase="Training",
                    loss=loss.item()
                )

            _val_loss, _val_score = validation(model, criterion, val_loader, DEVICE)
            _train_loss = np.mean(train_loss)
            print(f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Val Loss : [{_val_loss:.5f}] Val Weighted F1 Score : [{_val_score:.5f}]')

            scheduler.step()
                    
            if best_score < _val_score:
                best_score = _val_score
                best_model = model
        
    return best_model


def inference(model, test_loader, device):
    model.eval()
    preds = []
    le = preprocessing.LabelEncoder()
    with torch.no_grad():
        for imgs in tqdm((test_loader)):
            imgs = imgs.float().to(device)
            
            pred = model(imgs)
            
            preds += pred.argmax(1).detach().cpu().numpy().tolist()
    
    preds = le.inverse_transform(preds)
    return preds

if __name__ == "__main__":
    preds = main()

    submit = pd.read_csv('./sample_submission.csv')
    submit['label'] = preds
    submit.loc[submit['label'] == '0', 'label'] = '가구수정'
    submit.loc[submit['label'] == '1', 'label'] = '걸레받이수정'
    submit.loc[submit['label'] == '2', 'label'] = '곰팡이'
    submit.loc[submit['label'] == '3', 'label'] = '꼬임'
    submit.loc[submit['label'] == '4', 'label'] = '녹오염'
    submit.loc[submit['label'] == '5', 'label'] = '들뜸'
    submit.loc[submit['label'] == '6', 'label'] = '면불량'
    submit.loc[submit['label'] == '7', 'label'] = '몰딩수정'
    submit.loc[submit['label'] == '8', 'label'] = '반점'
    submit.loc[submit['label'] == '9', 'label'] = '석고수정'
    submit.loc[submit['label'] == '10', 'label'] = '오염'
    submit.loc[submit['label'] == '11', 'label'] = '오타공'
    submit.loc[submit['label'] == '12', 'label'] = '울음'
    submit.loc[submit['label'] == '13', 'label'] = '이음부불량'
    submit.loc[submit['label'] == '14', 'label'] = '창틀,문틀수정'
    submit.loc[submit['label'] == '15', 'label'] = '터짐'
    submit.loc[submit['label'] == '16', 'label'] = '틈새과다'
    submit.loc[submit['label'] == '17', 'label'] = '피스'
    submit.loc[submit['label'] == '18', 'label'] = '훼손'
