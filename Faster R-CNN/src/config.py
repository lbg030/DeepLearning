import torch

BATCH_SIZE = 8 # increase / decrease according to GPU memeory
RESIZE_TO = 512 # resize the image for training and transforms
NUM_EPOCHS = 1 # number of epochs to train for

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# training images and XML files directory
TRAIN_DIR = '../train_dataset/train'
# validation images and XML files directory
VALID_DIR = '../train_dataset/valid'

# classes: 0 index is reserved for background
CLASSES = [
    '양품', 'crack',
]

NUM_CLASSES = 2

# whether to visualize images after crearing the data loaders
VISUALIZE_TRANSFORMED_IMAGES = False

# location to save model and plots
OUT_DIR = '../outputs'
SAVE_PLOTS_EPOCH = 10 # save loss plots after these many epochs
SAVE_MODEL_EPOCH = 10 # save model after these many epochs