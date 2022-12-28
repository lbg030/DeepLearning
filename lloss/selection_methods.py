import numpy as np
import torch
from torch.utils.data import DataLoader

# Custom
from config import *
from data.sampler import SubsetSequentialSampler

def get_uncertainty(models, extractor, unlabeled_loader):
    models['backbone'].eval()
    models['module'].eval()
    uncertainty = torch.tensor([]).to(device)

    with torch.no_grad():
        for data in unlabeled_loader:
            inputs = data[0].to(device)
            
            features_dic = extractor(inputs)
            features = [features_dic['layer1'],features_dic['layer2'],features_dic['layer3'],features_dic['layer4']]
            
            pred_loss = models['module'](features) # pred_loss = criterion(scores, labels) # ground truth loss
            pred_loss = pred_loss.view(pred_loss.size(0))
            uncertainty = torch.cat((uncertainty, pred_loss), 0)
    
    return uncertainty.cpu()

# Select the indices of the unlablled data according to the methods
def query_samples(model,extractor, data_unlabeled, subset):

    # Create unlabeled dataloader for the unlabeled subset
    unlabeled_loader = DataLoader(data_unlabeled, batch_size=BATCH, 
                                sampler=SubsetSequentialSampler(subset), 
                                pin_memory=True)

    # Measure uncertainty of each data points in the subset
    uncertainty = get_uncertainty(model, extractor, unlabeled_loader)
    arg = np.argsort(uncertainty)        

    return arg
