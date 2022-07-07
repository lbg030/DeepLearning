import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init
from tqdm import tqdm

learning_rate = 1e-3
training_epochs = 5
batch_size = 64
mnist_train = dsets.MNIST(root='MNIST_data/', # 다운로드 경로 지정
 train=True, # True 를 지정하면 훈련 데이터로 다운로드
 transform=transforms.ToTensor(), # 텐서로 변환
 download=True)
mnist_test = dsets.MNIST(root='MNIST_data/', # 다운로드 경로 지정
 train=False, # False 를 지정하면 테스트 데이터로 다운로드
 transform=transforms.ToTensor(), # 텐서로 변환
 download=True)
data_loader = torch.utils.data.DataLoader(dataset=mnist_train,
 batch_size=batch_size,
shuffle=True,
 drop_last=True)
class RobustModel(torch.nn.Module):
 def __init__(self):
    super(RobustModel, self).__init__()
    self.keep_prob = 0.5
 # L1 ImgIn shape=(?, 28, 28, 1)
 # Conv -> (?, 28, 28, 32)
 # Pool -> (?, 14, 14, 32)
    self.layer1 = torch.nn.Sequential(
    torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2, stride=2))
 # L2 ImgIn shape=(?, 14, 14, 32)
 # Conv ->(?, 14, 14, 64)
 # Pool ->(?, 7, 7, 64)
    self.layer2 = torch.nn.Sequential(
    torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2, stride=2))
    # L3 ImgIn shape=(?, 7, 7, 64)
    # Conv ->(?, 7, 7, 128)
    # Pool ->(?, 4, 4, 128)
    self.layer3 = torch.nn.Sequential(
    torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1))
    # L4 FC 4x4x128 inputs -> 625 outputs
    self.fc1 = torch.nn.Linear(4 * 4 * 128, 625, bias=True)
    torch.nn.init.xavier_uniform_(self.fc1.weight)
    self.layer4 = torch.nn.Sequential(
    self.fc1,
    torch.nn.ReLU(),
    torch.nn.Dropout(p=1 - self.keep_prob))
    # L5 Final FC 625 inputs -> 10 outputs
    self.fc2 = torch.nn.Linear(625, 10, bias=True)
    torch.nn.init.xavier_uniform_(self.fc2.weight)
 def forward(self, x):
    out = self.layer1(x)
    out = self.layer2(out)
    out = self.layer3(out)
    out = out.view(out.size(0), -1)
    out = self.layer4(out)
    out = self.fc2(out)
    return out
# Robustmodel 모델 정의
model = RobustModel()
criterion = torch.nn.CrossEntropyLoss() # 비용 함수에 소프트맥스 함수 포함되어져 있음.
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
total_batch = len(data_loader)
# print('총 배치의 수 : {}'.format(total_batch))
# PATH='model.pt'
# for epoch in range(training_epochs):
#    avg_cost = 0
#    print(f"[Epoch {epoch+1} / {training_epochs}]")
#    for X, Y in tqdm(data_loader): # 미니 배치 단위로 꺼내온다. X 는 미니 배치, Y 느 ㄴ레이블.
#       optimizer.zero_grad()
      
#       hypothesis = model(X)
#       cost = criterion(hypothesis, Y)
#       cost.backward()
      
#       optimizer.step()
#       avg_cost += cost / total_batch
      
#    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))
   
#    torch.save(model.state_dict(), PATH)
with torch.no_grad():
   X_test = mnist_test.data.view(len(mnist_test), 1, 28, 28).float()
   Y_test = mnist_test.targets
   prediction = model(X_test)
   correct_prediction = torch.argmax(prediction, 1) == Y_test
   accuracy = correct_prediction.float().mean()
   print('Accuracy:', accuracy.item())