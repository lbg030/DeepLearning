from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F

from torchvision import transforms
import torchvision.transforms as T

from pathlib import Path
from PIL import Image
from config import *

data_src = "/Users/ibyeong-gwon/Downloads/ace_data"

defect_types = ['ac', 'as', 'ad', 'ah']
num_classes = len(defect_types)

defect_dict = {'ac':0,
                'as':1,
                'ad':2,
                'ah':3}


# 데이터 관련 함수 
class ACE(Dataset) : 
    def __init__(self, data_dir:str, training_type:str, transform) :
        super().__init__()
        assert training_type in ['train', 'test'], '데이터 타입을 다시 확인해주세요. train, valid, test 중 하나를 입력으로 넣어야합니다.'
        
        self.img_paths, self.labels = [], []
        self.transform = transform
    
        for path in Path(data_dir).glob(f'{training_type}/*/*') :
            if path.suffix in ['.jpg', '.png'] : 
                self.img_paths.append(str(path))
                self.labels.append(defect_dict[str(path.parent.name)])

    def __len__(self) : 
        return len(self.img_paths)
    
    def __getitem__(self, idx) :        
        img = Image.open(str(self.img_paths[idx]))
        labeled_img = str(self.img_paths[idx])
        
        label = self.labels[idx]

        if self.transform :
            img = self.transform(img)
            
        return img, label, labeled_img

train_transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.ToTensor(),
])

test_transform = T.Compose([
    T.ToTensor(),
])

train_data = ACE(data_dir=data_src, training_type='train', transform = train_transform)
test_data = ACE(data_dir=data_src, training_type='test', transform = test_transform)