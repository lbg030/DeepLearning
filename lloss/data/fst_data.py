# torch 관련 라이브러리 
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms as T

from pathlib import Path
from PIL import Image
from config import *

# config용 셀 

# 실험 관련 config 
n_exp = 19

# 데이터 관련 config 
# data_head = '/home/lbg030/luna/data/fst'
# data_src = "/home/lbg030/luna/data/experiments/experiment19"
data_src = "/home/lbg030/luna/data/custom_data"

defect_types = ['hz', 'bz', 'chem', 'yd']
num_classes = len(defect_types)
defect_dict = {'hz':0,
                'bz':1,
                'chem':2,
                'yd':3}

# model_save_dir = f'/home/lbg030/luna/data/experiments/experiment{n_exp}/results'
# metric_save_dir = f'/home/lbg030/luna/data/experiments/experiment{n_exp}/results'

# Path(model_save_dir).mkdir(parents=True, exist_ok=True)
# Path(metric_save_dir).mkdir(parents=True, exist_ok=True)

# 데이터 관련 함수 
class FSTData(Dataset) : 
    def __init__(self, data_dir:str, training_type:str, transform) :
        super().__init__()
        assert training_type in ['train', 'valid', 'test'], '데이터 타입을 다시 확인해주세요. train, valid, test 중 하나를 입력으로 넣어야합니다.'
        self.img_paths, self.labels = [], []
        self.transform = transform
        
        # if training_type == 'train':
        #     data_dir = "/home/lbg030/luna/data/fst"
        #     for path in Path(data_dir).glob(f'*/*') : 
        #         if path.suffix in ['.jpg', '.png'] : 
        #             self.img_paths.append(str(path))
        #             self.labels.append(defect_dict[str(path.parent.name)])
        # else :
        for path in Path(data_dir).glob(f'*/{training_type}/*') :
            if path.suffix in ['.jpg', '.png'] : 
                self.img_paths.append(str(path))
                self.labels.append(defect_dict[str(path.parent.parent.name)])

    def __len__(self) : 
        return len(self.img_paths)
    
    def __getitem__(self, idx) :
        # print(len(self.img_paths),len(self.labels), idx)
        
        img = Image.open(str(self.img_paths[idx]))
        label = self.labels[idx]

        if self.transform :
            img = self.transform(img)
        
        return img, label

# train_transform = T.Compose([
#         T.Resize((32,32)),
#         T.RandomHorizontalFlip(),
#         T.RandomCrop(size=32, padding=4),
#         T.ToTensor(),
#         T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]) # T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) # CIFAR-100
#     ])

# test_transform = T.Compose([
#         T.Resize((32,32)),
#         T.ToTensor(),
#         T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]) # T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) # CIFAR-100
#     ])

train_transform = T.Compose([
    # transforms.Resize((32, 32)),
    # transforms.Resize((224, 224)),
    transforms.Resize((360, 360)),
    T.RandomHorizontalFlip(),
    # T.RandomCrop(size=360, padding=4),
    
    T.ToTensor(),
    
    # T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]) # T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) # CIFAR-100
])

test_transform = T.Compose([
    # transforms.Resize((32, 32)),
    # transforms.Resize((224, 224)),
    transforms.Resize((360, 360)),
    T.ToTensor(),
    
    # T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]) # T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) # CIFAR-100
])

train_data = FSTData(data_dir=data_src, training_type='train', transform = train_transform)
unlabeled_data = FSTData(data_dir=data_src, training_type='train', transform = test_transform)
test_data = FSTData(data_dir=data_src, training_type='test', transform = test_transform)