import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import cv2
import glob
import pandas as pd
from ..src.config import *

all_img_list = glob.glob('/Users/gwonsair/Documents/dacon/data/train/*/*')
df = pd.DataFrame(columns=['img_path', 'label'])
df['img_path'] = all_img_list
df['label'] = df['img_path'].apply(lambda x : str(x).split('/')[-2])
train, val, _, _ = train_test_split(df, df['label'], test_size=0.3, stratify=df['label'], random_state=SEED)
le = preprocessing.LabelEncoder()
train['label'] = le.fit_transform(train['label'])
val['label'] = le.transform(val['label'])

class CustomDataset(Dataset):
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
    

train_transform =  A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Rotate(limit=15),
    A.Resize(640, 640),
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
    A.JpegCompression(p=0.1),
    A.CenterCrop(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])


test_transform = A.Compose([
                            A.Resize(640,640),
                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                            ToTensorV2()
                            ])

train_dataset = CustomDataset(train['img_path'].values, train['label'].values, train_transform)
train_loader = DataLoader(train_dataset, batch_size = BATCH, shuffle=False, num_workers=0)

val_dataset = CustomDataset(val['img_path'].values, val['label'].values, test_transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH, shuffle=False, num_workers=0)