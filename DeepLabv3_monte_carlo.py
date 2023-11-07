
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
torch.manual_seed(123)

class CustomDataset(Dataset):
        def __init__(self, csv_file, image_folder, target_folder,transform_image=None,transform_target=None):
            
            self.data = pd.read_csv(csv_file)
            
            
            #the scene input images and the target segmentation maps both have the same name but in different folder
            
            
            
            self.image_paths = self.data['file']
            self.image_paths = self.image_paths #for testing and debugging
            self.image_folder = image_folder
            self.target_folder = target_folder
            
            
            self.transform = transform
            self.transform_target = transform_target

        def __getitem__(self, index):
    

            
            # Construct the complete image path by joining the folder path and image name
            image_name = self.image_paths[index] 
            image_path = os.path.join(self.image_folder, image_name)

            
            # Open the image using PIL
            image = tiff.imread(image_path)
            #choos ethe number of channels you need
            image= image[:,:,:3]/10000
            

            target_path = os.path.join(self.target_folder, image_name)
            target = tiff.imread(target_path)
            
            
            
            
            

            if self.transform:
                
                image = self.transform(image)
                target = self.transform_target(target)
                
                
                
            return image, target

        def __len__(self):
            return len(self.image_paths)

torch.manual_seed(42)

# model = segmentation.deeplabv3_resnet50(pretrained=True)
# model.classifier[-1] = torch.nn.Conv2d(256, 43, kernel_size=1)
# checkpoint_path = r"C:\Users\midok\OneDrive\Desktop\Imam_farag_paper\checkpoint\pretrained_ep48_BS_4.pth"  # Replace with the actual path to your saved checkpoint file

# model.load_state_dict(torch.load(checkpoint_path))


# transform = transforms.Compose([
#             transforms.ToPILImage(),
            
#             transforms.Resize((512,512)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            
#         ])

model = segmentation.deeplabv3_resnet50(pretrained=True)
model.classifier[-1] = nn.Sequential(
nn.Conv2d(256, 43, kernel_size=1))

#checkpoint_path = r"C:\Users\midok\OneDrive\Desktop\Imam_farag_paper\checkpoint\allparm_CE_classindexedGT100.pth" # Replace with the actual path to your saved checkpoint file
checkpoint_path = 'allparm_CE_classindexedGT100.pth'
model.load_state_dict(torch.load(checkpoint_path))
                      
for param in model.parameters():
    param.requires_grad = False



# Make sure the last layer parameters are set to require gradients
# for param in model.classifier[-1].parameters():
#     param.requires_grad = True
    

model.eval()
model.to('cuda')

# Set up the optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

# Define the transformation for input images and labels
transform = transforms.Compose([
    
    transforms.ToTensor() 
])


transform_target = transforms.ToTensor()

csv_file =  '/data/home/shared/data/anthroprotect/infos.csv'
image_folder = '/data/home/shared/data/anthroprotect/tiles/s2'
target_folder='/data/home/aemam/ahmed_coding/greybox/new_masks/'

# Create the custom dataset
from torch.utils.data import DataLoader, random_split


# dataset = CustomDataset(csv_file, image_folder, target_folder,transform_image=transform)


# train_size = int(0.9955 * len(dataset))
# test_size = len(dataset) - train_size
# train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(123))

# # Create data loaders for the training and test subsets
# num_samples = 3
# batch_size=10


# train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
# test_dataloader = DataLoader(test_dataset, batch_size=batch_size , shuffle=False,drop_last=True)
# # Create the data loader
# dataloader = DataLoader(dataset, batch_size=batch_size , shuffle=True, drop_last=True)

# Create the custom dataset
dataset = CustomDataset(csv_file, image_folder, target_folder,transform_image=transform, transform_target =transform_target)
batch_size = 4
# Create the data loader

train_ratio = 0.8
test_ratio = 1 - train_ratio

# Calculate the number of samples for each split
train_size = int(train_ratio * len(dataset))
test_size = len(dataset) - train_size

# Perform the split
train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))

# Create data loaders for training and testing
dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True, generator=torch.Generator().manual_seed(42))
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last= True, generator=torch.Generator().manual_seed(42))





pretrained_model = model
input_data = dataset


model = model.to('cuda')




#---------------------------------------------------------------------
# For DeepLabvV3
# Define your Monte Carlo Dropout function
num_samples = 3

def mc_dropout(model, x, num_samples):
    torch.cuda.empty_cache()
    with torch.no_grad():
        model.train()
        x = x.to('cuda')
        #print(x.shape)
        #outputs = torch.zeros((num_samples,) + x.shape[1:])
        # smx = torch.softmax(model(x), dim=1)
        # print(smx.shape)
        # output_classes = torch.argmax(smx, 1)
        preds = [(torch.softmax(model(x)['out'], dim=1).cpu().detach()) for i in range(num_samples)]
        #pred = model(x)['out'].cpu().detach()
        #print('model output is -->',model(x)['out'].size)
        #preds.append(pred)
        #torch.argmax(torch.softmax(, dim=0), 0)
     
    return preds



# Create an empty tensor to store the outputs


# Create an empty tensor with the desired shape

all_predictions = []
print(len(test_dataloader))
# Loop over the input data
for imgs , target in tqdm(test_dataloader):
    imgs, target = imgs.cuda().float(), target.cuda().squeeze(1)
    batch_predictions = []
    pred_sets = mc_dropout(model, imgs, num_samples)
    img_predictions = torch.stack(pred_sets, dim=1)
    #batch_predictions.append(img_predictions)
    #batch_predictions.append(pred_sets)

    # imgs.cpu().detach()
    # target.cpu().detach()

  
  
    
    #batch_predictions = torch.stack(batch_predictions, dim=0)
    
    all_predictions.append(img_predictions)
    

    
monte_carlo_predictions = torch.cat(all_predictions, dim=0)
print('mc predcs--->',monte_carlo_predictions.shape)

#--------------------------------------------------------------------

var_img = monte_carlo_predictions[:, :, :, :, :].std(dim=1)
mean_var = var_img.mean(dim=(2,3))
#----------------------------------------------------------------
clcc = {111:'cont. urban fabric',
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

num_imgs = mean_var.size(0)
num_classes = mean_var.size(1)
std_classes = mean_var
class_names = clcc

def class_entropy(num_imgs, num_classes, class_names, std_classes):
    """Function to convet create a dictionary of each class

    Args:
        num_imgs (int): Nmber of input images.
        num_classes (int): Number of classes.
        class_names (dict): Dict of class names, the key is the class index, the value is the class name.

    Returns:
        dict: dict of class entropy, the key is the class index, the values are list with the first element being the class number and the second element being the entropy value.
    """


    # A data holder which contains the mean spatial pixel intensity of the entropy for each class across all the images we have
    class_pixel_mean_entropy = np.empty((num_imgs, num_classes))

    classes_dictionary = {}

    for i in range(num_imgs):
        for j in range(num_classes):
            # For image (i), class (j) ----> Get the mean pixel entropy intensity across the Height (H) and Width (W) of the image
            # Adding np.log2(num_classes) to normalize the entropy to be in range (0~1)
            class_pixel_mean_entropy[i,j] = (std_classes[i,j])/np.log2(2)

    
    for i, key in enumerate(class_names.keys()):
        # Convert the class_names dictionary values to list, in order to be able to add other values
        classes_dictionary[key] = [class_names.get(key)]

    for i, key in enumerate(classes_dictionary.keys()):
        # Append the value of the entropy per class across all images at the dictionary
        if len(classes_dictionary[key]) == 1:
            # The condition to avoid overwriting at the key
            # we are adding the class number and the mean pixel entropy intensity
            classes_dictionary[key].append([i ,class_pixel_mean_entropy[:, i].mean()])            
    
    return classes_dictionary

classes_dict = class_entropy(num_imgs, num_classes, clcc, mean_var)
print(classes_dict)

#----------------------------------------------------------------------------------------------------------#
np.save('classes_dict.npy', classes_dict)