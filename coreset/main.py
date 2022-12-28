'''
GCN Active Learning
'''

# Python
# import os
import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
# import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
# import torchvision.transforms as T
import torchvision.models as models
import argparse 
# Custom
# import models.resnet as resnet
# from models.resnet import vgg11
# from models.query_models import LossNet
from train_test import train, test
from load_dataset import load_dataset
from selection_methods import query_samples
from config import *
import timm
from pathlib import Path

# from data.fst_data import *

# import wandb
# wandb.init(project="classification", entity="lbg030")
# wandb.config = {
#   "learning_rate": LR,
#   "epochs": EPOCH,
#   "batch_size": BATCH
# }

random.seed(21)

# from plotly.subplots import make_subplots
# import plotly.graph_objects as go

parser = argparse.ArgumentParser()
parser.add_argument("-l","--lambda_loss",type=float, default=0.7, 
                    help="Adjustment graph loss parameter between the labeled and unlabeled")

parser.add_argument("-s","--s_margin", type=float, default=0.1,
                    help="Confidence margin of graph")

parser.add_argument("-n","--hidden_units", type=int, default=128,
                    help="Number of hidden units of the graph")

parser.add_argument("-r","--dropout_rate", type=float, default=0.3,
                    help="Dropout rate of the graph neural network")
parser.add_argument("-d","--dataset", type=str, default="fst",
                    help="")

parser.add_argument("-e","--no_of_epochs", type=int, default=EPOCH,
                    
                    help="Number of epochs for the active learner")
parser.add_argument("-m","--method_type", type=str, default="CoreSet",
                    help="")

parser.add_argument("-c","--cycles", type=int, default=CYCLES,
                    help="Number of active learning cycles")

parser.add_argument("-t","--total", type=bool, default=False,
                    help="Training on the entire dataset")

args = parser.parse_args()


##
# Main
if __name__ == '__main__':

    method = args.method_type
    methods = ['CoreSet']
    datasets = ['fst',]
    assert method in methods, 'No method %s! Try options %s'%(method, methods)
    assert args.dataset in datasets, 'No dataset %s! Try options %s'%(args.dataset, datasets)
    # Model - create new instance for every cycle so that it resets
    
    
            
    # results = open('results_'+str(args.method_type)+"_"+args.dataset +'_main'+str(args.cycles)+str(args.total)+'.txt','w')
    print("Dataset: %s"%args.dataset)
    print("Method type:%s"%method)
    if args.total:
        TRIALS = 1
        CYCLES = 1
        
    else:
        CYCLES = args.cycles
        
    for trial in range(TRIALS):
        # Load training and testing dataset
        data_train, data_unlabeled, data_test, adden, NO_CLASSES, no_train = load_dataset(args.dataset)
        # Don't predefine budget size. Configure it in the config.py: ADDENDUM = adden
        NUM_TRAIN = no_train
        indices = list(range(NUM_TRAIN))
        # print(trial, indices)
        random.shuffle(indices)

        with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                
                    #resnet18    = vgg11().to(device) 
                resnet18    =  timm.create_model('resnet50', num_classes = 7).to(device)

        models      = {'backbone': resnet18}
        torch.backends.cudnn.benchmark = True
        
        # optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion      = nn.CrossEntropyLoss()
        optim_backbone = optim.SGD(models['backbone'].parameters(), lr=LR, 
                momentum=MOMENTUM, weight_decay=WDECAY)

        sched_backbone = torch.optim.lr_scheduler.CosineAnnealingLR(optim_backbone, T_max=200)
        
        optimizers = {'backbone': optim_backbone}
        schedulers = {'backbone': sched_backbone}
            
        if args.total:
            labeled_set = indices
        else:
            labeled_set = indices[:ADDENDUM]
            unlabeled_set = [x for x in indices if x not in labeled_set]
        
        train_loader = DataLoader(data_train, batch_size=BATCH, num_workers=4, pin_memory=True, sampler=SubsetRandomSampler(labeled_set), drop_last = True)
        test_loader = DataLoader(data_test,num_workers=4, batch_size=1)
        
        dataloaders  = {'train': train_loader, 'test': test_loader}
        
        print(f"data_train = {len(data_train)},  data_test = {len(data_test)}")
        print(f"no_train = {NUM_TRAIN}")   
        
        for cycle in range(CYCLES):
            
            optim_backbone = optim.SGD(models['backbone'].parameters(), lr=LR, 
                momentum=MOMENTUM, weight_decay=WDECAY)
            
            # Randomly sample 10000 unlabeled data points
            if not args.total:
                random.shuffle(unlabeled_set)
                subset = unlabeled_set[:SUBSET]
            
            
    
            # Training and testing
            train_loss = train(models, method, criterion, optimizers, schedulers, dataloaders, args.no_of_epochs, EPOCHL)
            # print(f'train loss : {train_loss:.4f}')
            # metrics['train_loss'] = round(train_loss, 4)
            # train(models, method, criterion, optimizers, schedulers, dataloaders, args.no_of_epochs, EPOCHL)
            metrics, results = test(models, EPOCH, method, dataloaders, mode='test')
            print(f"accuracy : {metrics['accuracy']:.4f}")
            print(f"f1 score : {metrics['f1_score']:.4f}")
            print(f"precision : {metrics['precision']:.4f}")
            print(f"recall : {metrics['recall']:.4f}")
            
            print('Trial {}/{} || Cycle {}/{} || Label set size {}'.format(trial+1, TRIALS, cycle+1, CYCLES, len(labeled_set)))

            if cycle == (CYCLES-1):
                # Reached final training cycle
                print("Finished.")
                break
            
            if trial == 0 and cycle == 0:
                torch.save(models, str(Path(PATH, 'best.pt')))
                best_f1 = 0
                
            if best_f1 < metrics['f1_score']:
                best_f1 = metrics['f1_score']
                torch.save(models, str(Path(PATH, 'best.pt')))
                
            # Get the indices of the unlabeled samples to train on next cycle
            arg = query_samples(models, method, data_unlabeled, subset, labeled_set, cycle, args)
            
            # print(f"arg = {arg}")
            # Update the labeled dataset and the unlabeled dataset, respectively
            labeled_set += list(torch.tensor(subset)[arg][-ADDENDUM:].numpy())
            listd = list(torch.tensor(subset)[arg][:-ADDENDUM].numpy()) 
            unlabeled_set = listd + unlabeled_set[SUBSET:]

            dataloaders['train'] = DataLoader(data_train, batch_size=BATCH,sampler=SubsetRandomSampler(labeled_set),
                                            pin_memory=True,num_workers=4)