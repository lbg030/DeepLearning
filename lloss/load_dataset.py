import numpy as np
from torch.utils.data import DataLoader, Dataset
from config import *

from data.fst_data import *
# from data.aoi_data import *

class MyDataset(Dataset):
    def __init__(self, train_data):
        self.fst = train_data
            
    def __getitem__(self, index):
        data, target = self.fst[index]
        return data, target, index
    
    def __len__(self):
        return len(self.fst)


# Data
def load_dataset(dataset):
    data_train = train_data
    data_unlabeled = unlabeled_data
    data_test = test_data
    NO_CLASSES = 4
    adden = ADDENDUM
    # no_train = NUM_TRAIN
    no_train = len(train_data)
    
    print(f"train data : {len(train_data)}")
    return data_train, data_unlabeled, data_test, adden, NO_CLASSES, no_train