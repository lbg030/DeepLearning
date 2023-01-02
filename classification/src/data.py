from torch.utils.data import Dataset
import torchvision.transforms as T
from pathlib import Path
from PIL import Image
# from config import *

data_src = "C:/Users/lee97/Documents/GitHub/DeepLearning/data"

defect_type = ['dog', 'cat']

defect_dic = {
    'dog' : 0,
    'cat' : 1
}

class CLS(Dataset):
    def __init__(self, data_dir: str, training_type : str, transform):
        super().__init__() # TODO: super을 사용하는 이유 ?
        assert training_type in ['train','test'] , '데이터 타입 train / test 둘중 하나로 입력'
        self.img_paths, self.labels = [], []
        self.transform = transform
        
        for path in Path(data_dir).glob(f"*/{training_type}/*"):
            if path.suffix in ['.jpg', '.png']:
                self.img_paths.append(str(path))
                self.labels.append(defect_dic[str(path.parent.parent.name)])
                
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img = Image.open(str(self.img_paths[idx]))
        label = self.labels[idx]
        
        if self.transform:
            img = self.transform(img)
        return img,label
    
train_transform = T.Compose([
        T.Resize((224,224)),
        T.RandomHorizontalFlip(),
        T.ToTensor()
    ])

test_transform = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
    ])

train_data = CLS(data_dir=data_src, training_type='train', transform = train_transform)
test_data = CLS(data_dir=data_src, training_type='test', transform = test_transform)