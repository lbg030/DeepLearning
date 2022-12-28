# torch 관련 라이브러리 
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms as T

from pathlib import Path
from PIL import Image

# 데이터 관련 config 
data_src = "/home/lbg030/luna/data/aoi"

defect_types = ['0', '1','2','3','4','5','6']
num_classes = len(defect_types)
defect_dict = {'0':0,
                '1': 1,
                '2': 2,
                '3': 3,
                '4': 4,
                '5': 5,
                '6': 6,
                }

# 데이터 관련 함수 
class AOI(Dataset) : 
    def __init__(self, data_dir:str, training_type:str, transform) :
        super().__init__()
        assert training_type in ['train', 'valid', 'test'], '데이터 타입을 다시 확인해주세요. train, valid, test 중 하나를 입력으로 넣어야합니다.'
        self.img_paths, self.labels = [], []
        
        # self.labeled_paths = []
        
        self.transform = transform
    
        for path in Path(data_dir).glob(f'*/{training_type}/*') :
            if path.suffix in ['.jpg', '.png'] : 
                self.img_paths.append(str(path))
                self.labels.append(defect_dict[str(path.parent.parent.name)])

    def __len__(self) : 
        return len(self.img_paths)
    
    def __getitem__(self, idx) :
        # print(len(self.img_paths),len(self.labels), idx)
        
        img = Image.open(str(self.img_paths[idx]))
        # self.labeled_paths.append(self.img_paths[idx])
        labeled_img = str(self.img_paths[idx])
        
        label = self.labels[idx]

        if self.transform :
            img = self.transform(img)
            
        return img, label, labeled_img

train_transform = T.Compose([
    transforms.Resize((360, 360)),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
])

test_transform = T.Compose([
    transforms.Resize((360, 360)),
    T.ToTensor(),
])

train_data = AOI(data_dir=data_src, training_type='train', transform = train_transform)
unlabeled_data = AOI(data_dir=data_src, training_type='train', transform = test_transform)
test_data = AOI(data_dir=data_src, training_type='test', transform = test_transform)