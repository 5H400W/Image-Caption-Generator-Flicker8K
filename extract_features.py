"""
COMP5623M Coursework on Image Caption Generation


Forward pass through Flickr8k image data to extract and save features from
pretrained CNN.

"""


import torch
import numpy as np

import torch.nn as nn
from torchvision import transforms

from models import EncoderCNN
from datasets import Flickr8k_Images
from utils import *
from config import *



lines = read_lines(TOKEN_FILE_TRAIN)
# see what is in lines
# print(lines[:2])

#########################################################################
#
#       QUESTION 1.1 Text preparation
# 
#########################################################################

image_ids, cleaned_captions = parse_lines(lines)
# to check the results after writing the cleaning function
# print(image_ids[:2])
# print(cleaned_captions[:2])

# vocab = build_vocab(cleaned_captions)
# to check the results
# print("Number of words in vocab:", vocab.idx)


def cleaning_text(captions):
    table = str.maketrans('','',string.punctuation)
    for img,caps in captions.items():
        for i,img_caption in enumerate(caps):

            img_caption.replace("-"," ")
            desc = img_caption.split()

            #converts to lower case
            desc = [word.lower() for word in desc]
            #remove punctuation from each token
            desc = [word.translate(table) for word in desc]
            #remove hanging 's and a 
            desc = [word for word in desc if(len(word)>1)]
            #remove tokens with numbers in them
            desc = [word for word in desc if(word.isalpha())]
            #convert back to string

            img_caption = ' '.join(desc)
            captions[img][i]= img_caption
    return captions

def text_vocabulary(descriptions):
    # build vocabulary of all unique words
    vocab = set()
    
    for key in descriptions.keys():
        [vocab.update(d.split()) for d in descriptions[key]]
    
    return vocab

#All descriptions in one file 
def save_descriptions(descriptions, filename):
    lines = list()
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key + '\t' + desc )
    data = "\n".join(lines)
    file = open(filename,"w")
    file.write(data)
    file.close()



# crop size matches the input dimensions expected by the pre-trained ResNet
data_transform = transforms.Compose([ 
    transforms.Resize(224), 
    transforms.CenterCrop(224), 
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),   # using ImageNet norms
                         (0.229, 0.224, 0.225))])

dataset_train = Flickr8k_Images(
    image_ids=image_ids,
    image_dir=IMAGE_DIR,
    transform=data_transform,
)


train_loader = torch.utils.data.DataLoader(
    dataset_train,
    batch_size=64,
    shuffle=False,
    num_workers=2,
)

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EncoderCNN().to(device)



#########################################################################
#
#        QUESTION 1.2 Extracting image features
# 
#########################################################################

features = []

def extract_features(directory):
        model = Xception( include_top=False, pooling='avg' )
        features = {}
        for img in tqdm(os.listdir(directory)):
            filename = directory + "/" + img
            image = Image.open(filename)
            image = image.resize((299,299))
            image = np.expand_dims(image, axis=0)
            #image = preprocess_input(image)
            image = image/127.5
            image = image - 1.0
            
            feature = model.predict(image)
            features[img] = feature
        return features

#2048 feature vector
features = extract_features(dataset_images)
dump(features, open("features.p","wb"))


features = load(open("features.p","rb"))



# to check your results, features should be dimensions [len(train_set), 2048]
# convert features to a PyTorch Tensor before saving
# print(features.shape)



# save features 
torch.save(features, "features.pt")


