import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import cv2
import glob
import pandas as pd
import numpy as np
# from config import *
import warnings
warnings.filterwarnings("ignore")

data_src = "C:/Users/lee97/Documents/GitHub/DeepLearning/dacon/src/data"


class CustomDataset(Dataset):
    def __init__(self, data_dir:str, training_type:str, transform):
        assert training_type in ['train', 'valid', 'test']
        
        self.img_paths, self.labels = [], []
        self.transform = transform

        for path in Path(data_dir).glob(f'{training_type}/*/*') : 
            if path.suffix in ['.jpg', '.png'] : 
                self.img_paths.append(str(path))
                self.labels.append([str(path.parent.name)])
    def __len__(self) : 
        return len(self.img_paths)
    
    def __getitem__(self, idx) :
        # print(len(self.img_paths),len(self.labels), idx)
        
        img = Image.open(str(self.img_paths[idx]))
        img = np.array(img)

        label = self.labels[idx]

        if self.transform :
            img = self.transform(image=img)['image']

        return img, label
    

train_transform =  A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Rotate(limit=15),
    A.Resize(360, 360),
    A.OneOf([
        A.GaussianBlur(blur_limit=3, p=0.2),
        A.MotionBlur(blur_limit=3, p=0.2),
        A.MedianBlur(blur_limit=3, p=0.1)
    ], p=0.5),
    A.CLAHE(clip_limit=2.0, p=0.2),
    A.OneOf([
        A.RandomBrightness(p=0.5),
        A.RandomContrast(p=0.5),
        A.RandomGamma(p=0.5),
        A.HueSaturationValue(p=0.5)
    ], p=0.5),
    A.ISONoise(p=0.2),
    A.ImageCompression(p=0.1),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

class TestDataset(Dataset):
    def __init__(self, img_path_list, label_list, transforms=None):
        self.img_path_list = img_path_list
        self.label_list = label_list
        self.transforms = transforms
        
    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        
        image = cv2.imread(img_path)
        
        if self.transforms is not None:
            image = self.transforms(image=image)['image']
        
        if self.label_list is not None:
            label = self.label_list[index]
            return image, label
        else:
            return image
        
    def __len__(self):
        return len(self.img_path_list)
    

test_transform = A.Compose([
                            A.Resize(640,640),
                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                            ToTensorV2()
                            ])

train_dataset = CustomDataset(data_dir=data_src, training_type='train', transform = test_transform)
train_loader = DataLoader(train_dataset, batch_size = 16, shuffle=True, num_workers=4)

val_dataset = CustomDataset(data_dir=data_src, training_type='valid', transform = test_transform)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True, num_workers=4)

test = pd.read_csv('C:/Users/lee97/Documents/GitHub/DeepLearning/dacon/src/data/test.csv')
test_dataset = TestDataset(test['img_path'].values, None, test_transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=4)