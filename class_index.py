
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import segmentation
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import tifffile as tiff
import cv2
import pandas as pd
import tifffile as tiff
from tqdm import tqdm
import numpy as np
from PIL import Image
import os
import torch.nn as nn
import torch.nn.functional as F
class CustomDataset(Dataset):
            def __init__(self, csv_file, image_folder, target_folder,transform_image=None):
                
                self.data = pd.read_csv(csv_file)
                
                
                #the scene input images and the target segmentation maps both have the same name but in different folder
                
                
                
                self.image_paths = self.data['file']
                #self.image_paths = self.image_paths[:10] for testing and debugging
                self.image_folder = image_folder
                self.target_folder = target_folder
                
                
                self.transform = transform
 

            def __getitem__(self, index):
        

                
                # Construct the complete image path by joining the folder path and image name
                image_name = self.image_paths[index] 
                image_path = os.path.join(self.image_folder, image_name)

                
                # Open the image using PIL
                image = tiff.imread(image_path)
                #choos ethe number of channels you need
                image= image[:,:,:3]

                image = image.astype('uint8')
                
                #
                
                target_path = os.path.join(self.target_folder, image_name)
                target = tiff.imread(target_path)
                target = np.transpose(target, (1,2,0))





                target = cv2.resize(target, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)

                #target = target.astype('uint8')
                
                ## turning the ground truth segmenation mask into a (L,W,CLASSES) one hot-hot encoded matrix
                #target = Image.fromarray(target)
                #target = target.resize((512,512))
                
                
                

                if self.transform:
                    
                    image = self.transform(image)
  
                    
                    
                    

                return image, target

            def __len__(self):
                return len(self.image_paths)

######
# initialize dataset and data loaders
csv_file =  'infos.csv'
image_folder = '/home/ahmedemam576/working_folder/data/anthroprotect/tiles/s2'
target_folder='/home/ahmedemam576/greybox/corine_images/onehotcoded_masks/'

transform = transforms.Compose([
            transforms.ToPILImage(),
            
            transforms.Resize((512,512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            
        ])
# Create the custom dataset
from torch.utils.data import DataLoader, random_split


dataset = CustomDataset(csv_file, image_folder, target_folder,transform_image=transform)


'''train_size = int(0.9955 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create data loaders for the training and test subsets
num_samples = 3
batch_size=10


train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size , shuffle=False,drop_last=True)'''
# Create the data loader
dataloader = DataLoader(dataset, batch_size=20 , shuffle=False, drop_last=False)


#########
# change_onehotencoded to single channel class indexed ground truth
num_classes = 43

# Initialize an empty list to store the transformed labels
y_class_encoded = []

# For loop to iterate through the dataset and convert one-hot encoded labels to class encoded
for image, one_hot_label in tqdm(dataloader):
    one_hot_label = one_hot_label.permute(0, 3, 1, 2)
    
    # Find the class index by finding the position of 1 in the one-hot encoded label
    class_index = torch.argmax(one_hot_label, dim=1)
    
    # Append the class index to the list of class encoded labels
    y_class_encoded.append(class_index)
    np.save("stacked_images_tensor.pt", y_class_encoded)
# Convert the list of class indices into a tensor
# The shape of the tensor will be: (num_images, height, width)
np.save("stacked_images_tensor.pt", y_class_encoded)
stacked_tensor = torch.cat(y_class_encoded, dim=0)

# Save the stacked tensor to a file (e.g., a .pt or .pth file)
torch.save(stacked_tensor, )