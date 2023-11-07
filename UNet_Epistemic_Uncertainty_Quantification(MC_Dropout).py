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

train_ratio =  0.9
batch_size = 8



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
        self.image_paths = self.image_paths[0:100] #for testing and debugging only we selected the first 100 paths only to test that everything is working propely.
        # self.image_paths = self.image_paths[0:] # for loading the whole dataset
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

checkpoint_path = r"C:\Users\midok\OneDrive\Desktop\Imam_farag_paper\checkpoint\Unet\Unet_dropout_10in_43out_80.pth"

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
    """Function that creates a deconvolutional block for Unet architecture used at Decoder part

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.

    Returns:
        deconv (nn.Sequential): deconvblock moule that includes the structure of the block.
    """



    deconv = nn.Sequential(nn.Conv2d(in_channels= in_channels, out_channels= out_channels, kernel_size= (2,2), padding='same'),
                nn.BatchNorm2d(num_features=out_channels),
                nn.ReLU())
    
    return deconv

class unet(nn.Module):
    """ class for Unet architecture based on the original paper
    """

    def __init__(self, input_channels, output_classes, channel_list):
        """
        Args:
            input_channels (int): Number of input channels.
            output_classes (int): Number of output classes.
            channel_list (list): List of number of channels for each layer.
        """
        
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

        # self.architecture = f"""(512, 512, {self.channel_list[0]})....(512, 512, {self.channel_list[1]})...___________________________________________________________________________________________________________________________________________________________conv{self.channel_list[1]}__________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________..(concat{self.channel_list[1]})..(512, 512, {self.channel_list[2]})....(512, 512, {self.channel_list[1]})...(512, 512, {output_classes})
        #                                                                                                 .                                                                                                                                                                                                                                                                                                                                                                                                                                      .           
        #                                                                                                 .pooling                                                                                                                                                                                                                                                                                                                                                                                                                               . upsampling + deconv      
        #                                                                                                 .(256, 256, {self.channel_list[1]})....(256, 256, {self.channel_list[2]})..___________________________________________________________________________________conv{self.channel_list[2]}_________________________________________________________________________________________________________________________________________..(concat{self.channel_list[2]})..(256, 256, {self.channel_list[3]})....(256, 256, {self.channel_list[2]})...
        #                                                                                                                                                                         .                                                                                                                                                                                                                                                     .
        #                                                                                                                                                                         .pooling                                                                                                                                                                                                                                              .upsampling + deconv
        #                                                                                                                                                                         .(128, 128, {self.channel_list[2]})....(128, 128, {self.channel_list[3]})..___________conv{self.channel_list[3]}________________________________..(concat{self.channel_list[3]})..(128, 128, {self.channel_list[4]})....(128, 128, {self.channel_list[3]})...         
        #                                                                                                                                                                                                                                                 .                                                                    .    
        #                                                                                                                                                                                                                                                 .pooling                                                             .upsampling + deconv     
        #                                                                                                                                                                                                                                                 .(64, 64, {self.channel_list[3]})....(64, 64, {self.channel_list[4]})..."""


        

    def forward(self, x):
        """_summary_

        Args:
            x (torch.Tensor): input tensor.

        Returns:
            output (torch.Tensor): output segmentation tensor. 
        """

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
                       channel_list=channel_list, load_weights=True, weight_path=checkpoint_path)




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

    return model, optimizer, criterion

model, optimizer, criterion = training_settings(model, device='cuda', learning_rate=learning_rate, 
                                                optimizer_name='Adam', loss_function='cross_entropy')

#--------------------------------------Monte Carlo settings------------------------#

# Number of weights sampling
num_samples = 5
# Create an empty tensor to store the outputs
mc_outputs = torch.empty((batch_size,num_samples, 43, 256, 256))

def mc_dropout_one_batch(model, x, num_samples):

    """
    This function performs Monte Carlo Dropout for a single batch of data. 
    It generates multiple predictions for each sample in the batch by running the model with dropout enabled during inference. 
    The predictions are then stored for uncertainty quantification.

    Args:
        model (nn.Module): The model for which Monte Carlo Dropout is to be performed.
        x (torch.Tensor): The input data batch for which predictions are to be generated.
        num_samples (int): The number of Monte Carlo samples to be generated for each input sample.

    Returns:
        mc_outputs (torch.Tensor): A tensor containing the softmax outputs of the model for each Monte Carlo sample.
    """
    # Clear the GPU memory
    torch.cuda.empty_cache()

    # Ensure no gradients are computed for this operation
    with torch.no_grad():
        # Set all parameters to not require gradients
        for param in model.parameters():
             param.requires_grad = False

        # Set all Dropout layers to training mode
        for module in model.modules():
            if isinstance(module, nn.Dropout2d):
                module.train(True)
            
        # Move the input tensor to the GPU
        x = x.to('cuda')

        # Generate the Monte Carlo samples
        for i in range(num_samples):
            # Compute the softmax of the model output and store it in the mc_outputs tensor
            mc_outputs[:,i, :, :, :] = torch.softmax(model(x), dim=1).cpu().detach()
     
    return mc_outputs

# create empty list to save the predictions
all_predictions = []
loader = test_dataloader



def mc_dropout_all_batches(loader, model, num_samples):
    """
    This function performs Monte Carlo Dropout for all batches of data. 
    It generates multiple predictions for each sample in each batch by running the model with dropout enabled during inference. 
    The predictions are then stored for uncertainty quantification.

    Args:
        loader (DataLoader): The DataLoader object that provides batches of data for inference.
        model (nn.Module): The model for which Monte Carlo Dropout is to be performed.
        num_samples (int): The number of Monte Carlo samples to be generated for each input sample.

    Returns:
        all_predictions (list): A list containing the softmax outputs of the model for each Monte Carlo sample for all batches.
    """

    # Iterate over all batches provided by the DataLoader
    for imgs , target in tqdm(loader):
        # Move the images and targets to the GPU and convert the targets to the required shape
        imgs, target = imgs.cuda().float(), target.cuda().squeeze(1)
        
        # Perform Monte Carlo Dropout for this batch and get the predictions
        pred_sets = mc_dropout_one_batch(model, imgs, num_samples)
        
        # Store the predictions in the img_predictions tensor
        img_predictions = pred_sets
        
        # Append the predictions for this batch to the list of all predictions
        all_predictions.append(img_predictions)
        
        # Delete the variables to reduce memory consumption
        del pred_sets
        del img_predictions
        del imgs
    
    # Return the list of all predictions
    return all_predictions

all_predictions = mc_dropout_all_batches(loader, model, num_samples)

monte_carlo_predictions = torch.cat(all_predictions, dim=0)

#print function to check the shape -> it should be (batch size, num_samples, output_classes, heigth, width)
print('\n\n The shape of the outputs is:---------->',monte_carlo_predictions.shape)

#---------------------------------------Calculating the standard deviation-------------------------#
"""The main idea here, for each class we can assess the uncertainty by first check the standard deviation between the different runs made by Monte Carlo,
then we can sum the standard deviation values for each class spatially. 

* So if there is variation between different tuns -> example four pixels have value = [1, 1, 1, 1] then their mean would be also 1 and the standard deviation would be zero! in this case,
we can say that the uncertainty is too low, otherwise we will have different mean and different standard deviation values, and we add a final step by adding all of the pixel values for each class,
if the model is uncertain we could say that there will be standard deviation betweem runs, hence the spatial summation will show that."""

std_img = monte_carlo_predictions[:, :, :, :, :].std(dim=1)
sum_std = std_img.sum(dim=(2,3))

def scale_to_minus_one_one(x, min_value, max_value):
    """
    This function scales a given value (x) to the range of -1 to 1 using the provided minimum and maximum values.

    Args:
        x (float): The value to be scaled.
        min_value (float): The minimum value in the original scale.
        max_value (float): The maximum value in the original scale.

    Returns:
        scaled_x (float): The value of x scaled to the range of -1 to 1.
    """
    # Compute the scaled value
    scaled_x = -1 + 2 * (x - min_value) / (max_value - min_value)
    
    return scaled_x

# Assign each code to the class name
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

num_imgs = sum_std.size(0)
num_classes = sum_std.size(1)
std_classes = sum_std
class_names = clcc

def class_std(num_imgs, num_classes, class_names, sum_variance_values):
    """
    Function to create a dictionary of each class with its mean standard deviation. to show the standard deviation of the class presented between different runs of Monte Carlo.
    Then we take mean across t

    Args:
        num_imgs (int): Number of input images.
        num_classes (int): Number of classes.
        class_names (dict): Dict of class names, the key is the class index, the value is the class name.
        variance_values (numpy.ndarray): 2D array with variance values for each class in each image.

    Returns:
        dict: Dictionary where the key is the class index, and the values are a list with the first element being the class name and the second element being the mean standard deviation value.
    """

    # Initialize an empty numpy array to store the mean standard deviation for each class across all images
    class_pixel_mean_std = np.empty((num_imgs, num_classes))

    # Initialize an empty dictionary to store the results
    classes_dictionary = {}

    # Loop over each image and class
    for i in range(num_imgs):
        for j in range(num_classes):
            # Scale the variance value to be between -1 and 1 and store it in the array
            class_pixel_mean_std[i,j] = (scale_to_minus_one_one(sum_variance_values[i,j], min(sum_variance_values[i,:]), max(sum_variance_values[i,:])))

    # Loop over each class
    for i, key in enumerate(class_names.keys()):
        # Add the class name to the dictionary
        classes_dictionary[key] = [class_names.get(key)]

    # Loop over each class again
    for i, key in enumerate(classes_dictionary.keys()):
        # If the class only has the name in the dictionary (length of value list is 1)
        if len(classes_dictionary[key]) == 1:
            # Append the class index and the mean standard deviation to the dictionary
            classes_dictionary[key].append([i ,class_pixel_mean_std[:, i].mean()])            
    
    return classes_dictionary

classes_dict = class_std(num_imgs, num_classes, clcc, sum_std)

print(classes_dict)

# if you want to save this dictionary to be used later to calculate the CNE metric
# np.save('classes_dict.npy', classes_dict)

#_________________________________________________________________