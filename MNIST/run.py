import argparse
import os
from skimage import color
from skimage import io
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from model import RobustModel
import numpy as np

class ImageDataset(Dataset):
    """ Image shape: 28x28x3 """
    
    def __init__(self, root_dir, fmt=':06d', extension='.png'):
        self.root_dir = root_dir
        self.fmtstr = '{' + fmt + '}' + extension

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.fmtstr.format(idx)
        img_path = os.path.join(self.root_dir, img_name)
        img = plt.imread(img_path)
        data = torch.from_numpy(img)
        data = color.rgb2gray(data)
        data = np.expand_dims(data,0)
        return data


def inference(data_loader, model):
    """ model inference """

    model.eval()
    preds = []

    with torch.no_grad():
        for X in data_loader:
            y_hat = model(X)
            y_hat.argmax()

            _, predicted = torch.max(y_hat, 1)
            preds.extend(map(lambda t: t.item(), predicted))

    return preds


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='2022 DL Term Project #1')
    parser.add_argument('--load_model', default='model.pt', help="Model's state_dict")
    parser.add_argument('--dataset', default='./test/', help='image dataset directory')
    parser.add_argument('--batch_size', type=int, default=16, help='test loader batch size')

    args = parser.parse_args()

    # instantiate model
    model = RobustModel()
    model.load_state_dict(torch.load(args.load_model))

    # load dataset in test image folder
    test_data = ImageDataset(args.dataset)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size)

    # write model inference
    preds = inference(test_loader, model)

    with open('result.txt', 'w') as f:
        f.writelines('\n'.join(map(str, preds)))
