#final code
# to do list is to build a transparent model to train on the classes
import glob
import os


import pandas as pd
from PIL import Image

import time

import torch
import torchvision.transforms as transforms

from torchvision.models import segmentation
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import os
import tifffile as tiff
import cv2
import pandas as pd

from tqdm import tqdm
import numpy as np

import os
import torch.nn as nn
import torch.nn.functional as F




class CustomDataset(Dataset):
        def __init__(self, csv_file, image_folder,transform_image=None):
            
            self.data = pd.read_csv(csv_file)
            
            
            #the scene input images and the target segmentation maps both have the same name but in different folder
            
            
            
            self.image_paths = self.data['file']
            self.image_paths = self.image_paths[:] #for testing and debugging
            self.image_folder = image_folder
            self.labels = self.data['label']
            
            
            self.transform = transform
            

        def __getitem__(self, index):
    

            
            # Construct the complete image path by joining the folder path and image name
            image_name = self.image_paths[index] 
            image_path = os.path.join(self.image_folder, image_name)

            
            # Open the image using PIL
            image = tiff.imread(image_path)
            #choos ethe number of channels you need
            image= image[:,:,:3]/10000
            

            
            image_label = self.labels[index]
            
            
            

            if self.transform:
                
                image = self.transform(image)
                
            return image, image_label

        def __len__(self):
            return len(self.image_paths)
        #-----------------------------------------------------------------------------------#

    # Define the paths to your training data


    # Set random seed for reproducibility
torch.manual_seed(42)

    # Initialize the model and set it to training mode

model = segmentation.deeplabv3_resnet50(pretrained=True)
model.classifier[-1] = nn.Sequential(
nn.Conv2d(256, 43, kernel_size=1))
PATH= 'allparm_CE_classindexedGT100.pth'
model.load_state_dict(torch.load(PATH))
model.eval()



# Make sure the last layer parameters are set to require gradients
'''for param in model.classifier[-1].parameters():
    param.requires_grad = True
    '''

model.train()
model.to('cuda')

# Set up the optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

# Define the transformation for input images and labels
transform = transforms.Compose([
    
    transforms.ToTensor() 
])



#---------------------------------------------------------------------------------------------------#

csv_file =  'infos.csv'
image_folder = '/home/ahmedemam576/working_folder/data/anthroprotect/tiles/s2'
target_folder='/home/ahmedemam576/greybox/corine_images/new_masks/'
    
# Create the custom dataset
dataset = CustomDataset(csv_file, image_folder,transform_image=transform)
batch_size = 40
# Create the data loader

train_ratio = 0.8
test_ratio = 1 - train_ratio

# Calculate the number of samples for each split
train_size = int(train_ratio * len(dataset))
test_size = len(dataset) - train_size

# Perform the split
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create data loaders for training and testing
dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last= True)


# Path to the CSV file
csv_file = 'infos.csv'

# Create an instance of the custom dataset
dataset = CustomDataset(csv_file,'/home/ahmedemam576/working_folder/data/anthroprotect/tiles/s2')

# Create a data loader to iterate over the dataset
batch_size = 1
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
##################################

torch.manual_seed(42)

    # Initialize the model and set it to training mode



model.eval()
model.to('cuda')













clcc ={111:'cont. urban fabric',
 112:'disc urban fabric',
 121:'industrial or commercial units',
 122:'road and rail',
 123:'port areas',
 124:'airports',
 131:'mineral extraction sites',
 132:'dump sites',
 133:'construction sites',
 141:'green urban areas',
 142:'sport and leasure',
 211:'non irregated arable land',
 212:'permenant irregated land',
 213:'rice fields',
 221:'vine yards',
 223:'olive groves',
 231:'pastures',
 241:'annual with perm. crops',
 242:'complex cultivation patters',
 243:'land principally occupied by agriculture',
 244:'agro forest areas',
 311:'broad leaved forest',
 312:'conferous forest',
 313:'mixed forest',
 321:'natural grassland',
 322:'moors and heathland',
 323: 'scierohllous vegitation',
 324:'transitional woodland shrub',
 331: 'beaches dunes and sand plains',
 332:'bare rock',
 333:'sparsely vegetated areas',
 334:'burnt areas',
 335:'glaciers and perpetual snow',
 411:'inland marshes',
 412:'peat bogs',
 421:'salt marshes',
 422:'salines',
 423:'intertidal flats',
 511:'water courses',
 512:'water bodies',
 521:'costal lagoons',
 522:'estuaries',
 523:'sea and ocean'}
 
landcover_corine = [key for idx, key in enumerate(clcc) ]
landcovers = [idx for idx, key in enumerate(clcc) ]
land_cover_classes_ordered = landcovers
feature_array_list = []
target_labels_list = []

# Create a list of land cover classes in the order of clcc dictionary
with torch.no_grad():
    for images, labels in tqdm(test_dataloader):
        # Move data to the same device as the model (CPU or GPU)
        images = images.cuda().float()

        # Get the predictions using the model
        outputs = model(images)['out']
        probs = torch.sigmoid(outputs)
        print('probs after sigmoid',outputs.shape)
        # Convert the outputs to a segmentation mask
        segmentation_masks = outputs.argmax(dim=1)
        print('probs after argmax',segmentation_masks.shape)
        print('labels',labels.shape)
        for segmentation_mask in segmentation_masks:
            # Initialize the dictionary to store the pixel counts for each class
            class_counts = {key: 0 for key in land_cover_classes_ordered}

            # Count the pixels for each land cover class in the segmentation_mask
            for class_key in torch.unique(segmentation_mask):
                class_rep = torch.sum(segmentation_mask == class_key)
                class_counts[class_key.item()] = class_rep.item()

            # Append the counts to the feature_array_list
            feature_array_list.append(list(class_counts.values()))

        # Append the target labels to target_labels_list
        target_labels_list.append(labels.numpy())

# Convert feature_array_list and target_labels_list to NumPy arrays
feature_array_list = np.array(feature_array_list)

# Calculate the correct number of batches and images per batch
num_batches = len(test_dataloader)
images_per_batch = feature_array_list.shape[0] // num_batches
num_classes = feature_array_list.shape[1]

# Reshape feature_array_list to add the batch size dimension
feature_array = feature_array_list.reshape(num_batches, images_per_batch, num_classes)

# Flatten target_labels_list to ensure it matches the number of images in the dataset
target_labels = np.array(target_labels_list)

# Reshape target_labels_list to add the batch size dimension
target_labels = target_labels.reshape(num_batches, images_per_batch)

# Save feature_array and target_labels as numpy arrays
np.save('feature_array.npy', feature_array)
np.save('target_labels.npy', target_labels)
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
    
'''for images,  labels  in tqdm(test_dataloader):
print('inside the dataloader loop')
# Convert images to segmentation mask here using your own logic
images = images.cuda().float()


outputs = model(images)['out']

segmentation_mask = outputs
#print(len(images))
# Get the unique land cover classes present in the segmentation mask
land_cover_classes = torch.unique(segmentation_mask)

# Compute the number of land cover classes and the maximum number of classes expected
num_classes = len(land_cover_classes)
max_num_classes = 43  # Set the maximum number of land cover classes expected in the dataset
 # an array with numbers starting with zero and ends with 43

# create a dictionary to have the key the land cover class code and as the ni. of pixels of this specific land cover class as data
class_counts = {}



# Generate the vector with the count of pixels for each land cover class

class_counts = {}  # Initialize class_counts outside the loop

for class_key in land_cover_classes:
    print('inside land cover classes loop')
    if class_key in landcovers:
        print('inside if statement')# Move the if condition outside the loop
        class_counts[class_key.item()] = 0  # Initialize count for the current class_key to 0
        print(class_counts)
        class_rep = torch.sum(segmentation_mask == class_key)
        class_counts[class_key.item()] = class_rep.item()



feature_array.append(list(class_counts.values()))
target_labels.append(labels.item())
feature_array = np.array(feature_array)
target_labels = np.array(target_labels)

# Save feature_array and target_labels as numpy arrays
np.save('feature_array.npy', feature_array)
np.save('target_labels.npy', target_labels)'''