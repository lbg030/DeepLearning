import torch

BATCH_SIZE = 16 # increase / decrease according to GPU memeory
RESIZE_TO = 224 # resize the image for training and transforms
NUM_EPOCHS = 1 # number of epochs to train for

DEVICE = torch.device('cuda:3') if torch.cuda.is_available() else torch.device('cpu')

# training images and XML files directory
TRAIN_DIR = '/Users/ibyeong-gwon/Downloads/train'
# validation images and XML files directory
VALID_DIR = '/Users/ibyeong-gwon/Downloads/test'

# classes: 0 index is reserved for background
CLASSES = [
    "0","1","2","3","4","5"
]

NUM_CLASSES = 6

# whether to visualize images after crearing the data loaders
VISUALIZE_TRANSFORMED_IMAGES = False

# location to save model and plots
OUT_DIR = '/Users/ibyeong-gwon/Desktop/Git/DeepLearning/Faster R-CNN/output'
SAVE_PLOTS_EPOCH = 10 # save loss plots after these many epochs
SAVE_MODEL_EPOCH = 10 # save model after these many epochs