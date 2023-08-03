if __name__ == "__main__":
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
    from PIL import Image
    import os
    import torch.nn as nn
    import torch.nn.functional as F


    
    
    
    class CustomDataset(Dataset):
        def __init__(self, csv_file, image_folder, target_folder,transform_image=None,transform_target=None):
            
            self.data = pd.read_csv(csv_file)
            
            
            #the scene input images and the target segmentation maps both have the same name but in different folder
            
            
            
            self.image_paths = self.data['file']
            self.image_paths = self.image_paths[:] #for testing and debugging
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
        #-----------------------------------------------------------------------------------#

    # Define the paths to your training data


    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Initialize the model and set it to training mode

    model = segmentation.deeplabv3_resnet50(pretrained=True)
    model.classifier[-1] = nn.Sequential(
    nn.Conv2d(256, 43, kernel_size=1))
    for param in model.parameters():
        param.requires_grad = True



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


    transform_target = transforms.ToTensor()

    #---------------------------------------------------------------------------------------------------#

    csv_file =  'infos.csv'
    image_folder = '/home/ahmedemam576/working_folder/data/anthroprotect/tiles/s2'
    target_folder='/home/ahmedemam576/greybox/corine_images/new_masks/'
        
    # Create the custom dataset
    dataset = CustomDataset(csv_file, image_folder, target_folder,transform_image=transform, transform_target =transform_target)
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





    # Training loop
    num_epochs = 100
    for epoch in tqdm(range(num_epochs), desc="Main training"):
        avgloss = 0
        for images, targets in tqdm(dataloader, desc="Current epoch", leave = False):
            model.train()
            optimizer.zero_grad()
            images, targets = images.cuda().float(), targets.cuda().squeeze(1)
            
            
            outputs = model(images)  
            
            loss = criterion(outputs['out'], targets)
            avgloss += loss.detach().cpu()
            # Backward pass and optimization
            loss.backward()


            optimizer.step()
            
        avgloss= avgloss/len(train_dataset)   
        print(f'avg loss for epoch{epoch+1} ={avgloss}', end='\n')
        if epoch+1%2:
            
            torch.save(model.state_dict(), f'allparm_CE_classindexedGT{epoch+1}.pth')

        
            



