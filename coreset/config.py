''' Configuration File.
'''

device = 'cuda:2'
CUDA_VISIBLE_DEVICES = 2

NUM_TRAIN = 9836 # N 트레인 데이터 수
BATCH     = 8 # B 
SUBSET    = 2000 # M
ADDENDUM  = 100 # K
TRIALS = 5
CYCLES = 5


# 몇개씩 라벨 셋 사이즈를 설정할건지 ( 몇개씩 증가할 건지 -> a number of data per ex) # 상관 관계를 잘 모름
PATH = "/home/lbg030/luna/workspaces/Intern/coreset/models/model"

MARGIN = 1.0 # xi

WEIGHT = 1.0 # lambda

EPOCH = 200

# EPOCH_GCN = 20
LR = 0.1

MILESTONES = [160, 240]

EPOCHV = 100 # VAAL number of epochs
EPOCHL = 120#20 #120 # After 120 epochs, stop 

MOMENTUM = 0.9
WDECAY = 5e-4#2e-3# 5e-4