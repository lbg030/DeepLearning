# torch 관련 라이브러리 
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms as T
import os
from pathlib import Path
from PIL import Image

# 데이터 관련 config 
# data_head = '/home/lbg030/luna/data/fst'
data_src = "/Users/gwonsmpro/Desktop/Pillip Img"

defect_types = os.listdir(data_src)
defect_types = [x for x in defect_types if os.path.isdir(data_src+"/"+x)]

print(defect_types)
num_classes = len(defect_types)
defect_dict = {defect: int(i) for i,defect in enumerate(defect_types)}

# 데이터 관련 함수 
class TSNE(Dataset) : 
    def __init__(self, data_dir:str , transform) :
        super().__init__()
        self.img_paths, self.labels = [], []
        self.transform = transform

        for path in Path(data_dir).glob(f'*/*') : 
            if path.suffix in ['.jpg', '.png', '.bmp'] : 
                self.img_paths.append(str(path))
                self.labels.append(defect_dict[str(path.parent.name)])

    def __len__(self) : 
        return len(self.img_paths)
    
    def __getitem__(self, idx) :

        img = Image.open(str(self.img_paths[idx])).convert('L')
        img = img.convert('RGB')
        label = self.labels[idx]

        if self.transform :
            img = self.transform(img)

        return img, label

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

tsne_data = TSNE(data_dir=data_src, transform = transform)