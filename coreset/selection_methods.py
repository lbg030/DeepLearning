import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
# Custom
from config import *
from models.query_models import VAE, Discriminator, GCN
from data.sampler import SubsetSequentialSampler
from kcenterGreedy import kCenterGreedy

def BCEAdjLoss(scores, lbl, nlbl, l_adj):
    lnl = torch.log(scores[lbl])
    lnu = torch.log(1 - scores[nlbl])
    labeled_score = torch.mean(lnl) 
    unlabeled_score = torch.mean(lnu)
    bce_adj_loss = -labeled_score - l_adj*unlabeled_score
    return bce_adj_loss


def aff_to_adj(x, y=None):
    x = x.detach().cpu().numpy()
    adj = np.matmul(x, x.transpose())
    adj +=  -1.0*np.eye(adj.shape[0])
    adj_diag = np.sum(adj, axis=0) #rowise sum
    adj = np.matmul(adj, np.diag(1/adj_diag))
    adj = adj + np.eye(adj.shape[0])
    adj = torch.Tensor(adj).to(device)

    return adj

def read_data(dataloader, labels=True):
    if labels:
        while True:
            for img, label,_ in dataloader:
                yield img, label
    else:
        while True:
            for img, _, _ in dataloader:
                yield img

def vae_loss(x, recon, mu, logvar, beta):
    mse_loss = nn.MSELoss()
    MSE = mse_loss(recon, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD = KLD * beta
    return MSE + KLD

def train_vaal(models, optimizers, labeled_dataloader, unlabeled_dataloader, cycle):
    
    vae = models['vae']
    discriminator = models['discriminator']
    vae.train()
    discriminator.train()
    with torch.cuda.device(CUDA_VISIBLE_DEVICES):
        vae = vae.to(device)
        discriminator = discriminator.to(device)
    
    adversary_param = 1
    beta          = 1
    num_adv_steps = 1
    num_vae_steps = 2

    bce_loss = nn.BCELoss()
    
    labeled_data = read_data(labeled_dataloader)
    unlabeled_data = read_data(unlabeled_dataloader)

    train_iterations = int( (ADDENDUM*cycle+ SUBSET) * EPOCHV / BATCH )

    for iter_count in range(train_iterations):
        labeled_imgs, labels = next(labeled_data)
        unlabeled_imgs = next(unlabeled_data)[0]

        with torch.cuda.device(CUDA_VISIBLE_DEVICES):
            labeled_imgs = labeled_imgs.to(device)
            unlabeled_imgs = unlabeled_imgs.to(device)
            labels = labels.to(device)

        # VAE step
        for count in range(num_vae_steps): # num_vae_steps
            recon, _, mu, logvar = vae(labeled_imgs)
            unsup_loss = vae_loss(labeled_imgs, recon, mu, logvar, beta)
            unlab_recon, _, unlab_mu, unlab_logvar = vae(unlabeled_imgs)
            transductive_loss = vae_loss(unlabeled_imgs, 
                    unlab_recon, unlab_mu, unlab_logvar, beta)
        
            labeled_preds = discriminator(mu)
            unlabeled_preds = discriminator(unlab_mu)
            
            lab_real_preds = torch.ones(labeled_imgs.size(0))
            unlab_real_preds = torch.ones(unlabeled_imgs.size(0))
                
            with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                lab_real_preds = lab_real_preds.to(device)
                unlab_real_preds = unlab_real_preds.to(device)

            dsc_loss = bce_loss(labeled_preds[:,0], lab_real_preds) + \
                       bce_loss(unlabeled_preds[:,0], unlab_real_preds)
            total_vae_loss = unsup_loss + transductive_loss + adversary_param * dsc_loss
            
            optimizers['vae'].zero_grad()
            total_vae_loss.backward()
            optimizers['vae'].step()

            # sample new batch if needed to train the adversarial network
            if count < (num_vae_steps - 1):
                labeled_imgs, _ = next(labeled_data)
                unlabeled_imgs = next(unlabeled_data)[0]

                with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                    labeled_imgs = labeled_imgs.to(device)
                    unlabeled_imgs = unlabeled_imgs.to(device)
                    labels = labels.to(device)

        # Discriminator step
        for count in range(num_adv_steps):
            with torch.no_grad():
                _, _, mu, _ = vae(labeled_imgs)
                _, _, unlab_mu, _ = vae(unlabeled_imgs)
            
            labeled_preds = discriminator(mu)
            unlabeled_preds = discriminator(unlab_mu)
            
            lab_real_preds = torch.ones(labeled_imgs.size(0))
            unlab_fake_preds = torch.zeros(unlabeled_imgs.size(0))

            with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                lab_real_preds = lab_real_preds.to(device)
                unlab_fake_preds = unlab_fake_preds.to(device)
            
            dsc_loss = bce_loss(labeled_preds[:,0], lab_real_preds) + \
                       bce_loss(unlabeled_preds[:,0], unlab_fake_preds)

            optimizers['discriminator'].zero_grad()
            dsc_loss.backward()
            optimizers['discriminator'].step()

            # sample new batch if needed to train the adversarial network
            if count < (num_adv_steps-1):
                labeled_imgs, _ = next(labeled_data)
                unlabeled_imgs = next(unlabeled_data)[0]

                with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                    labeled_imgs = labeled_imgs.to(device)
                    unlabeled_imgs = unlabeled_imgs.to(device)
                    labels = labels.to(device)
            if iter_count % 100 == 0:
                print("Iteration: " + str(iter_count) + "  vae_loss: " + str(total_vae_loss.item()) + " dsc_loss: " +str(dsc_loss.item()))
                
def get_uncertainty(models, unlabeled_loader):
    models['backbone'].eval()
    models['module'].eval()
    with torch.cuda.device(CUDA_VISIBLE_DEVICES):
        uncertainty = torch.tensor([]).to(device)

    with torch.no_grad():
        for inputs, _, _ in unlabeled_loader:
            with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                inputs = inputs.to(device)
            _, _, features = models['backbone'](inputs)
            pred_loss = models['module'](features) # pred_loss = criterion(scores, labels) # ground truth loss
            pred_loss = pred_loss.view(pred_loss.size(0))
            uncertainty = torch.cat((uncertainty, pred_loss), 0)
    
    return uncertainty.cpu()

def get_features(models, unlabeled_loader):
    models['backbone'].eval()
    with torch.cuda.device(CUDA_VISIBLE_DEVICES):
        features = torch.tensor([]).to(device)    
    with torch.no_grad():
            for inputs, _, _ in unlabeled_loader:
                with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                    inputs = inputs.to(device)
                    _, features_batch, _ = models['backbone'](inputs)
                features = torch.cat((features, features_batch), 0)
            feat = features #.detach().cpu().numpy()
    return feat

def get_kcg(models, labeled_data_size, unlabeled_loader):
    models['backbone'].eval()
    with torch.cuda.device(CUDA_VISIBLE_DEVICES):
        features = torch.tensor([]).to(device)

    with torch.no_grad():
        for data in unlabeled_loader:
            with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                inputs = data[0].to(device)
            features_batch= models['backbone'](inputs)
            features = torch.cat((features, features_batch), 0)
        feat = features.detach().cpu().numpy()
        new_av_idx = np.arange(SUBSET,(SUBSET + labeled_data_size))
        sampling = kCenterGreedy(feat)  
        batch = sampling.select_batch_(new_av_idx, ADDENDUM)
        other_idx = [x for x in range(SUBSET) if x not in batch]
        
    return  other_idx + batch


# Select the indices of the unlablled data according to the methods
def query_samples(model, method, data_unlabeled, subset, labeled_set, cycle, args):
    
    if method == 'CoreSet':
        # Create unlabeled dataloader for the unlabeled subset
        unlabeled_loader = DataLoader(data_unlabeled, batch_size=BATCH, 
                                    # sampler=SubsetSequentialSampler(subset), # more convenient if we maintain the order of subset
                                    sampler=SubsetSequentialSampler(subset+labeled_set), # more convenient if we maintain the order of subset
                                    pin_memory=True)

        arg = get_kcg(model, ADDENDUM*(cycle+1), unlabeled_loader)

    return arg
