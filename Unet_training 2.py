import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import segmentation
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import os
import tifffile as tiff
import cv2
import pandas as pd
import tifffile as tiff
from tqdm import tqdm
import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as F


seed = 78
torch.manual_seed(seed)
np.random.seed(seed)

# Import necessary libraries and modules for the Unet implementation.
# These include torch, torchvision, PIL, pandas, os, tifffile, cv2, tqdm, and numpy.
# Additionally, set the random seed to 78 using torch.manual_seed(78).


#-------------------------------Data Settings---------------------------------#

csv_file =  r"C:\Users\midok\OneDrive\Desktop\Imam_farag_paper\anthroprotect\infos.csv"
image_folder = r"C:\Users\midok\OneDrive\Desktop\Imam_farag_paper\anthroprotect\tiles\s2"
target_folder= r"C:\Users\midok\OneDrive\Desktop\Imam_farag_paper\new_masks"

train_ratio = 0.8
batch_size = 12

def transformations():
    """
    This function creates a transformation pipeline for the input images. 
    The pipeline consists of a single operation, which is converting the input image to a PyTorch tensor.

    Returns:
        tuple: A tuple containing the transformation pipeline for the input images and the target images.
    """
    # Define the transformation pipeline for the input images.
    # The pipeline consists of a single operation, which is converting the input image to a PyTorch tensor.
    transform = transforms.Compose([
        transforms.ToTensor() 
    ])

    # Define the transformation pipeline for the target images.
    # In this case, we don't apply any transformations to the target images, so we just return the same tensor.
    transform_target = transforms.ToTensor()

    return transform, transform_target

transform, transform_target = transformations()

class CustomDataset(Dataset):
    
    def __init__(self, csv_file, image_folder, target_folder,transform_image=None,transform_target=None):
        """ 
        Initialize the CustomDataset object.

        Args:
            csv_file (str): Path to the CSV file containing the image file names.
            image_folder (str): Path to the folder containing the input images.
            target_folder (str): Path to the folder containing the target segmentation maps.
            transform_image (torchvision.transforms.Compose, optional): Transformations to be applied on the input images. Defaults to None.
            transform_target (torchvision.transforms.Compose, optional): Transformations to be applied on the target segmentation maps. Defaults to None.
        """
        
        self.data = pd.read_csv(csv_file)
        
        
        #the scene input images and the target segmentation maps both have the same name but in different folder
        
        
        
        self.image_paths = self.data['file']
        # self.image_paths = self.image_paths[0:100] #for testing and debugging only we selected the first 100 paths only to test that everything is working propely.
        self.image_paths = self.image_paths[0:] # for loading the whole dataset
        self.image_folder = image_folder
        self.target_folder = target_folder
        
        
        self.transform = transform
        self.transform_target = transform_target

    def __getitem__(self, index):
        """
        Fetches an image and its corresponding target at the given index from the dataset.

        Args:
            index (int): Index of the image and target to be fetched.

        Returns:
            image (torch.Tensor): The input image at the given index after applying the necessary transformations.
            target (torch.Tensor): The corresponding target of the input image at the given index after applying the necessary transformations.
        """


        
        # Construct the complete image path by joining the folder path and image name
        image_name = self.image_paths[index] 
        image_path = os.path.join(self.image_folder, image_name)

        
        # Open the image using PIL and Open the image using tifffile library and normalize it by dividing by 10000 (for normalization)
        image = tiff.imread(image_path)
        
        # print(image_path)
        #choose the number of channels you want - here we selected the whole 10 channels
        image= image[:,:,:]/10000
        
        # Open the target using tifffile library
        target_path = os.path.join(self.target_folder, image_name)
        target = tiff.imread(target_path)
        
        
        
        # Apply the transformations to the image and target if any
        if self.transform:
            
            image = self.transform(image)
            target = self.transform_target(target)
            
            
            
        return image, target

    def __len__(self):
        return len(self.image_paths)


dataset_all = CustomDataset(csv_file, image_folder, target_folder,
                            transform_image=transform, transform_target =transform_target)

def create_dataloder(dataset_all, train_ratio, batch_size, seed=seed):
    """
    This function creates a dataloader for the training and testing datasets.

    Args:
        dataset_all (Dataset): The complete dataset which needs to be split into training and testing datasets.
        train_ratio (float): The ratio of the total dataset to be used for training. Should be between 0 and 1.
        batch_size (int): The number of samples per batch to load.
        seed (int, optional): The seed for the random number generator. Defaults to seed.

    Returns:
        tuple: A tuple containing the dataloader for the training dataset and the dataloader for the testing dataset.
    """
    # Calculate the number of samples for each split
    train_size = int(train_ratio * len(dataset_all))
    test_size = len(dataset_all) - train_size

    # Perform the split
    # Here we ensure that the dataset each time it will be loaded will have the same data distribution between training and test
    train_dataset, test_dataset = random_split(dataset_all, [train_size, test_size], generator=torch.Generator().manual_seed(seed))

    # Create dataloaders for the training and testing datasets
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last= True)
    
    return dataloader, test_dataloader 

dataloader, test_dataloader = create_dataloder(dataset_all, train_ratio=train_ratio, batch_size=batch_size, seed=seed)

#--------------------------------Model Settings--------------------------------#

# Unet (10-channels input and 43 channels outputs)
input_channels = 10
output_classes = 43
channel_list = [input_channels, 64, 128, 256, 512]


def convblock(in_channels, out_channels, kernel_size = (3,3), padding='same'):
    """Convolutional block for Unet architecture used at Encoder and Decoder parts
        
        :param in_channels: number of input channels.
        :param out_channels: number of output channels.
        :param kernel_size: kernel size of the convolution operation, default = (3,3).
        :param padding: padding operation, default = 'same'.
        
        :return conv: module that includes the structure of the block, type=nn.Sequential()"""
    
    
    conv = nn.Sequential(nn.Conv2d(in_channels= in_channels, out_channels= out_channels, kernel_size= kernel_size, padding=padding, device='cuda'),
                nn.BatchNorm2d(num_features=out_channels, device='cuda'),
                nn.ReLU(),
                nn.Conv2d(in_channels= out_channels, out_channels= out_channels, kernel_size= kernel_size, padding=padding, device='cuda'),
                nn.BatchNorm2d(num_features=out_channels, device='cuda'),
                nn.ReLU(),
                )
    
    return conv

def deconvblock(in_channels, out_channels):

    """De-Convolutional block for Unet architecture used at Encoder and Decoder parts
        
        :param in_channels: number of input channels.
        :param out_channels: number of output channels.

        
        :return deconv: module that includes the structure of the block, type = nn.Sequential()"""

    deconv = nn.Sequential(nn.Conv2d(in_channels= in_channels, out_channels= out_channels, kernel_size= (2,2), padding='same'),
                nn.BatchNorm2d(num_features=out_channels),
                nn.ReLU())
    
    return deconv

class unet(nn.Module):
    

    def __init__(self, input_channels, output_classes, channel_list):
        # call the parent constructor
        super(unet, self).__init__()

        # Encoder Layers
        self.channel_list = channel_list

        self.conv_blocks =  [convblock(self.channel_list[i], self.channel_list[i+1]) if i != len(self.channel_list) -1 else  None for i in range(len(self.channel_list))]
        # self.convblock64  = convblock(in_channels=input_channels, out_channels=64)
        # self.convblock128 = convblock(in_channels=64, out_channels=128)
        # self.convblock256 = convblock(in_channels=128, out_channels=256)
        # self.convblock512 = convblock(in_channels=256, out_channels=512)

        # Maxpooling layer
        self.maxpool2D = nn.MaxPool2d((2,2))

        # Upsampling layer
        self.upsampling2D = nn.Upsample(scale_factor=2, mode='bilinear')

        # Dropout Layer
        self.drop2D = nn.Dropout2d(p = 0.2)

        
        # Decoder layers

        self.deconvblock256 = deconvblock(in_channels=512, out_channels=256)
        self.convblock256_decoder = convblock(in_channels=512, out_channels=256)

        self.deconvblock128 = deconvblock(in_channels=256, out_channels=128)
        self.convblock128_decoder = convblock(in_channels=256, out_channels=128)

        self.deconvblock64 = deconvblock(in_channels=128, out_channels=64)
        self.convblock64_decoder = convblock(in_channels=128, out_channels=64)

        # self.deconvblock3 = deconvblock(in_channels=64, out_channels=43)

        # Output layer

        self.output = nn.Conv2d(in_channels=64, out_channels=output_classes, kernel_size=(1,1))

        self.architecture = f"""(512, 512, {self.channel_list[0]})....(512, 512, {self.channel_list[1]})...___________________________________________________________________________________________________________________________________________________________conv{self.channel_list[1]}__________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________..(concat{self.channel_list[1]})..(512, 512, {self.channel_list[2]})....(512, 512, {self.channel_list[1]})...(512, 512, {output_classes})
                                                                                                        .                                                                                                                                                                                                                                                                                                                                                                                                                                      .           
                                                                                                        .pooling                                                                                                                                                                                                                                                                                                                                                                                                                               . upsampling + deconv      
                                                                                                        .(256, 256, {self.channel_list[1]})....(256, 256, {self.channel_list[2]})..___________________________________________________________________________________conv{self.channel_list[2]}_________________________________________________________________________________________________________________________________________..(concat{self.channel_list[2]})..(256, 256, {self.channel_list[3]})....(256, 256, {self.channel_list[2]})...
                                                                                                                                                                                .                                                                                                                                                                                                                                                     .
                                                                                                                                                                                .pooling                                                                                                                                                                                                                                              .upsampling + deconv
                                                                                                                                                                                .(128, 128, {self.channel_list[2]})....(128, 128, {self.channel_list[3]})..___________conv{self.channel_list[3]}________________________________..(concat{self.channel_list[3]})..(128, 128, {self.channel_list[4]})....(128, 128, {self.channel_list[3]})...         
                                                                                                                                                                                                                                                        .                                                                    .    
                                                                                                                                                                                                                                                        .pooling                                                             .upsampling + deconv     
                                                                                                                                                                                                                                                        .(64, 64, {self.channel_list[3]})....(64, 64, {self.channel_list[4]})..."""


        

    def forward(self, x):
        """Forward pass of the Unet architecture
        
        :param x: input tensor, type = torch.FloatTensor()
        
        :return x: output tensor, type = torch.FloatTensor()"""

        # Encoder part

        # conv64 = self.convblock64(x)
        conv64 = self.conv_blocks[0](x)
        pool64 = self.maxpool2D(conv64)
        # Dimensions: (512, 512, 3) ---> convblock64 ---> (512, 512, 64) ---> maxpool2D ---> (256, 256, 64)

        # conv128 = self.convblock128(pool64)
        conv128 = self.conv_blocks[1](pool64)
        pool128 = self.maxpool2D(conv128)
        # Dimensions: (256, 256, 64) ---> convblock128 ---> (256, 256, 128) ---> maxpool2D ---> (128, 128, 128)

        # conv256 = self.convblock256(pool128)
        conv256 = self.conv_blocks[2](pool128)
        conv256 = self.drop2D(conv256)
        pool256 = self.maxpool2D(conv256)
        # Dimensions: (128, 128, 128) ---> convblock128 ---> (128, 128, 256) ---> maxpool2D ---> (64, 64, 256)

        # conv512 = self.convblock512(pool256)
        conv512 = self.conv_blocks[3](pool256)
        conv512 = self.drop2D(conv512)
        # Dimensions: (64, 64, 256) ---> convblock128 ---> (64, 64, 512)

    
        # Decoder part

        up256 = self.upsampling2D(conv512)
        deconv256 = self.deconvblock256(up256)
        # Dimensions: (64, 64, 512) ---> upsampling2D ---> (128, 128, 512) ---> deconvblock256 ---> (128, 128, 256)
        concat256 = torch.concatenate([deconv256, conv256], dim=1)
        concat256 = self.convblock256_decoder(concat256)
        # Dimensions: (128, 128, 256) ---> concatenate ---> (128, 128, 512) ---> convblock256_decoder ---> (128, 128, 256)

        up128 = self.upsampling2D(concat256)
        deconv128 = self.deconvblock128(up128)
        # Dimensions: (128, 128, 256) ---> upsampling2D ---> (256, 256, 256) ---> deconvblock128 ---> (256, 256, 128)
        concat128 = torch.concat([deconv128, conv128], dim=1)
        concat128 = self.convblock128_decoder(concat128)
        # Dimensions: (256, 256, 128) ---> concatenate ---> (256, 256, 256) ---> convblock128_decoder ---> (256, 256, 128)


        up64 = self.upsampling2D(concat128)
        deconv64 = self.deconvblock64(up64)
        # Dimensions: (256, 256, 128) ---> upsampling2D ---> (512, 512, 128) ---> deconvblock64 ---> (512, 512, 64)
        concat64 = torch.concat([deconv64, conv64], dim=1)
        concat64 = self.convblock64_decoder(concat64)
        # Dimensions: (512, 512, 64) ---> concatenate ---> (512, 512, 128) ---> convblock64_decoder ---> (512, 512, 64)

        output = self.output(concat64)
        # Dimensions: (512, 512, 64) ---> output ---> (512, 512, 43)
        
        return output
    
def model_settings(input_channels, output_classes, channel_list, load_weights=False, weight_path=None):
    """
    This function initializes a UNet model with the specified parameters. 
    If load_weights is set to True, it loads the model weights from the specified path.

    Args:
        input_channels (int): The number of input channels for the UNet model.
        output_classes (int): The number of output classes for the UNet model.
        channel_list (list): List of number of channels for each layer in the UNet model.
        load_weights (bool, optional): If True, load model weights from the specified path. Defaults to False.
        weight_path (str, optional): The path to the model weights file. Required if load_weights is True.

    Returns:
        model (nn.Module): The initialized UNet model.
    """
    # Initialize the UNet model with the specified parameters
    model = unet(input_channels=input_channels, output_classes=output_classes, channel_list=channel_list) 

    # If load_weights is True, load the model weights from the specified path
    if load_weights:
        checkpoint_path = weight_path
        model.load_state_dict(torch.load(checkpoint_path))

    return model               

model = model_settings(input_channels=input_channels, output_classes=output_classes, 
                       channel_list=channel_list, load_weights=False)




#-----------------------------------Training settings-----------------------------#

learning_rate = 0.001
optimizer_name = 'Adam'
loss_function = 'cross_entropy'

def training_settings(model, device="cuda", learning_rate=0.001, optimizer_name = 'Adam', loss_function = 'cross_entropy'):
    """
    This function sets up the model for training by moving it to the specified device, 
    initializing the optimizer with the specified learning rate, and setting up the loss function.

    Args:
        model (nn.Module): The model to be set up for training.
        device (str, optional): The device to which the model should be moved. Defaults to "cuda".
        learning_rate (float, optional): The learning rate for the optimizer. Defaults to 0.001.
        optimizer_name (str, optional): The name of the optimizer to be used. Defaults to 'Adam'.
        loss_function (str, optional): The name of the loss function to be used. Defaults to 'cross_entropy'.
    
    Returns:
        model (nn.Module): The model moved to the specified device.
        optimizer (torch.optim): The initialized optimizer.
        criterion (torch.nn.Module): The loss function.
    """
    
    # Move the model to the specified device
    if device == 'cuda':
        model.cuda()
    else:
        model.cpu()

    # Initialize the optimizer with the specified learning rate
    if optimizer_name == 'Adam':      
        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Set up the loss function
    if loss_function == 'cross_entropy':
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError('Please specify a valid loss function.')

    #Put the model to the training state
    model.train()

    return model, optimizer, criterion

model, optimizer, criterion = training_settings(model, device='cuda', learning_rate=learning_rate, 
                                                optimizer_name='Adam', loss_function='cross_entropy')

#-------------------------------Training loop---------------------------------
# Here we only train the model without validation dataset checking

num_epochs = 100
loader = dataloader

def model_train(model, loader, num_epochs=100, save=False):
    """
    This function trains the model for a specified number of epochs.

    Args:
        model (nn.Module): The model to be trained.
        loader (DataLoader): The DataLoader object containing the training data.
        num_epochs (int, optional): The number of epochs for which the model should be trained. Defaults to 100.
        save (bool, optional): If True, save the model weights after each epoch. Defaults to False.
    """

    # Iterate over each epoch
    for epoch in tqdm(range(num_epochs), desc="Main training"):

        # Iterate over each batch in the DataLoader
        for images, targets in tqdm(loader, desc="Current epoch", leave = False):
            # Zero the gradients
            optimizer.zero_grad()

            # Move the images and targets to the GPU and adjust the dimensions
            images, targets = images.cuda().float(), targets.cuda().squeeze(1)
            
            # Forward pass: compute the outputs of the model given the images
            outputs = model(images)
            
            # Compute the loss between the outputs and the targets
            loss = criterion(outputs, targets)
            
            # Backward pass: compute the gradients of the loss w.r.t. the model's parameters
            loss.backward()

            # Update the model's parameters
            optimizer.step()
            
        # If save is True, save the model weights after each epoch
        if save:    
            if (epoch+1) % 2:
                torch.save(model.state_dict(), f'C:/Users/midok/OneDrive/Desktop/Imam_farag_paper/checkpoint/Unet/Unet_dropout_10in_43out_{epoch+1}.pth')      

model_train(model, loader, num_epochs=100, save=False)

#-----