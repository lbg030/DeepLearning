import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

BATCH = 8 # B 
EPOCH = 200
LR = 0.1