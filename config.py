"""
Parameters. You will need to customize the directories to match your
environment.

"""


##################################################################
# 
#         		DIRECTORIES
# 
################################################################### 


# path to caption data
CAPTION_DIR = ""
# path to images
IMAGE_DIR = ""

# token file names to read from
TOKEN_FILE_TRAIN = CAPTION_DIR + "Flickr8k_train.token.txt"
TOKEN_FILE_TEST = CAPTION_DIR + "Flickr8k_test.token.txt"


##################################################################
# 
#         TRAINING and VOCAB PARAMS (defaults - you may change)
# 
################################################################### 

MIN_FREQUENCY = 3

EMBED_SIZE = 256
HIDDEN_SIZE = 512
NUM_LAYERS = 1
LR = 0.001
NUM_EPOCHS = 5
LOG_STEP = 10


