"""
Naturalness Assessment using Deep Learning, XAI and Uncertainty Quantification

This script implements a framework for assessing naturalness of landscapes using:
1. Semantic segmentation with DeepLabV3 or UNet
2. Uncertainty quantification via Monte Carlo dropout
3. XAI through surrogate modeling with logistic regression

The approach extracts land cover class distributions from semantic segmentation outputs,
quantifies model uncertainty using Monte Carlo dropout, and identifies influential
features through a surrogate logistic regression model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.models import segmentation

import numpy as np
import pandas as pd
import os
import tifffile as tiff
from tqdm import tqdm
import cv2
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# Set random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# CORINE Land Cover classes
CORINE_CLASSES = {
    111: 'cont. urban fabric',
    112: 'disc urban fabric',
    121: 'industrial or commercial units',
    122: 'road and rail',
    123: 'port areas',
    124: 'airports',
    131: 'mineral extraction sites',
    132: 'dump sites',
    133: 'construction sites',
    141: 'green urban areas',
    142: 'sport and leisure',
    211: 'non irrigated arable land',
    212: 'permanent irrigated land',
    213: 'rice fields',
    221: 'vine yards',
    223: 'olive groves',
    231: 'pastures',
    241: 'annual with perm. crops',
    242: 'complex cultivation patterns',
    243: 'land principally occupied by agriculture',
    244: 'agro forest areas',
    311: 'broad leaved forest',
    312: 'coniferous forest',
    313: 'mixed forest',
    321: 'natural grassland',
    322: 'moors and heathland',
    323: 'sclerophyllous vegetation',
    324: 'transitional woodland shrub',
    331: 'beaches dunes and sand plains',
    332: 'bare rock',
    333: 'sparsely vegetated areas',
    334: 'burnt areas',
    335: 'glaciers and perpetual snow',
    411: 'inland marshes',
    412: 'peat bogs',
    421: 'salt marshes',
    422: 'salines',
    423: 'intertidal flats',
    511: 'water courses',
    512: 'water bodies',
    521: 'coastal lagoons',
    522: 'estuaries',
    523: 'sea and ocean'
}

# Data loading and processing
class CustomDataset(Dataset):
    """
    Dataset class for loading satellite images and their corresponding segmentation masks.
    """
    def __init__(self, csv_file, image_folder, target_folder=None, transform_image=None, transform_target=None):
        """
        Initialize the CustomDataset.
        
        Args:
            csv_file (str): Path to the CSV file containing image information
            image_folder (str): Path to the folder containing the input images
            target_folder (str, optional): Path to the folder containing target segmentation masks
            transform_image (callable, optional): Transformations to apply to the input images
            transform_target (callable, optional): Transformations to apply to the target masks
        """
        self.data = pd.read_csv(csv_file)
        self.image_paths = self.data['file']
        self.image_folder = image_folder
        self.target_folder = target_folder
        self.transform = transform_image
        self.transform_target = transform_target
        
        # If target_folder is None, assume we're only using images and their labels
        if target_folder is None and 'label' in self.data.columns:
            self.labels = self.data['label']
            self.inference_mode = True
        else:
            self.inference_mode = False

    def __getitem__(self, index):
        """
        Get an item from the dataset.
        
        Args:
            index (int): Index of the item to get
            
        Returns:
            tuple: (image, target) or (image, label) depending on the mode
        """
        # Get image path and load image
        image_name = self.image_paths[index]
        image_path = os.path.join(self.image_folder, image_name)
        
        # Load image and normalize (divide by 10000 for Sentinel-2 data)
        image = tiff.imread(image_path)
        
        # For DeepLabV3, use only RGB channels (first 3)
        # For UNet, can use all channels
        image = image[:, :, :3] / 10000
        
        # Apply image transformations if any
        if self.transform:
            image = self.transform(image)
        
        # If in inference mode, return image and label
        if self.inference_mode:
            return image, self.labels[index]
        
        # Otherwise, return image and target mask
        target_path = os.path.join(self.target_folder, image_name)
        target = tiff.imread(target_path)
        
        if self.transform_target:
            target = self.transform_target(target)
            
        return image, target

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.image_paths)


def create_dataloaders(dataset, train_ratio, batch_size, seed=SEED):
    """
    Create train and test dataloaders from a dataset.
    
    Args:
        dataset (Dataset): The dataset to split
        train_ratio (float): Ratio of data for training
        batch_size (int): Batch size
        seed (int): Random seed for reproducibility
        
    Returns:
        tuple: (train_dataloader, test_dataloader)
    """
    # Calculate split sizes
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    
    # Split dataset
    train_dataset, test_dataset = random_split(
        dataset, 
        [train_size, test_size], 
        generator=torch.Generator().manual_seed(seed)
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True
    )
    
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        drop_last=True
    )
    
    return train_dataloader, test_dataloader


def create_transformations():
    """
    Create transformation pipelines for the input images and targets.
    
    Returns:
        tuple: (transform, transform_target)
    """
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    transform_target = transforms.ToTensor()
    
    return transform, transform_target


# Model definitions and utilities
class UNet(nn.Module):
    """
    U-Net architecture for semantic segmentation with dropout for uncertainty estimation.
    """
    
    def __init__(self, input_channels, output_classes, channel_list):
        """
        Initialize the UNet model.
        
        Args:
            input_channels (int): Number of input channels
            output_classes (int): Number of output classes
            channel_list (list): List of channel counts for each layer
        """
        super(UNet, self).__init__()
        
        self.channel_list = channel_list
        
        # Encoder blocks
        self.conv_block1 = self._conv_block(channel_list[0], channel_list[1])
        self.conv_block2 = self._conv_block(channel_list[1], channel_list[2])
        self.conv_block3 = self._conv_block(channel_list[2], channel_list[3])
        self.conv_block4 = self._conv_block(channel_list[3], channel_list[4])
        
        # Pooling and dropout
        self.maxpool = nn.MaxPool2d((2, 2))
        self.dropout = nn.Dropout2d(p=0.2)
        
        # Decoder blocks
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        
        self.deconv_block3 = self._deconv_block(channel_list[4], channel_list[3])
        self.conv_block3_decoder = self._conv_block(channel_list[3] * 2, channel_list[3])
        
        self.deconv_block2 = self._deconv_block(channel_list[3], channel_list[2])
        self.conv_block2_decoder = self._conv_block(channel_list[2] * 2, channel_list[2])
        
        self.deconv_block1 = self._deconv_block(channel_list[2], channel_list[1])
        self.conv_block1_decoder = self._conv_block(channel_list[1] * 2, channel_list[1])
        
        # Output layer
        self.output = nn.Conv2d(channel_list[1], output_classes, kernel_size=(1, 1))
        
    def _conv_block(self, in_channels, out_channels, kernel_size=(3, 3), padding='same'):
        """Helper method to create a convolutional block."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
    def _deconv_block(self, in_channels, out_channels):
        """Helper method to create a deconvolutional block."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(2, 2), padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        """Forward pass of the UNet model."""
        # Encoder path
        conv1 = self.conv_block1(x)
        pool1 = self.maxpool(conv1)
        
        conv2 = self.conv_block2(pool1)
        pool2 = self.maxpool(conv2)
        
        conv3 = self.conv_block3(pool2)
        conv3 = self.dropout(conv3)
        pool3 = self.maxpool(conv3)
        
        conv4 = self.conv_block4(pool3)
        conv4 = self.dropout(conv4)
        
        # Decoder path
        up3 = self.upsample(conv4)
        deconv3 = self.deconv_block3(up3)
        concat3 = torch.cat([deconv3, conv3], dim=1)
        conv3_d = self.conv_block3_decoder(concat3)
        
        up2 = self.upsample(conv3_d)
        deconv2 = self.deconv_block2(up2)
        concat2 = torch.cat([deconv2, conv2], dim=1)
        conv2_d = self.conv_block2_decoder(concat2)
        
        up1 = self.upsample(conv2_d)
        deconv1 = self.deconv_block1(up1)
        concat1 = torch.cat([deconv1, conv1], dim=1)
        conv1_d = self.conv_block1_decoder(concat1)
        
        output = self.output(conv1_d)
        
        return output


def setup_model(model_type, input_channels, output_classes, device='cuda', checkpoint_path=None):
    """
    Set up a semantic segmentation model.
    
    Args:
        model_type (str): Type of model ('deeplabv3' or 'unet')
        input_channels (int): Number of input channels
        output_classes (int): Number of output classes
        device (str): Device to use ('cuda' or 'cpu')
        checkpoint_path (str, optional): Path to model checkpoint
        
    Returns:
        model: The initialized model
    """
    if model_type.lower() == 'deeplabv3':
        model = segmentation.deeplabv3_resnet50(pretrained=True)
        model.classifier[-1] = nn.Sequential(
            nn.Conv2d(256, output_classes, kernel_size=1)
        )
    elif model_type.lower() == 'unet':
        channel_list = [input_channels, 64, 128, 256, 512]
        model = UNet(input_channels, output_classes, channel_list)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Load checkpoint if provided
    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path))
    
    # Set model to evaluation mode and move to device
    model.eval()
    model = model.to(device)
    
    return model


def train_model(model, train_dataloader, test_dataloader, num_epochs=100, 
               learning_rate=0.001, device='cuda', save_path=None):
    """
    Train a semantic segmentation model.
    
    Args:
        model: The model to train
        train_dataloader: DataLoader for training data
        test_dataloader: DataLoader for test data
        num_epochs (int): Number of training epochs
        learning_rate (float): Learning rate
        device (str): Device to use ('cuda' or 'cpu')
        save_path (str, optional): Path to save model checkpoints
        
    Returns:
        model: The trained model
    """
    model = model.to(device)
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in tqdm(range(num_epochs), desc="Training"):
        avg_loss = 0
        
        for images, targets in tqdm(train_dataloader, desc=f"Epoch {epoch+1}", leave=False):
            images = images.to(device).float()
            targets = targets.to(device).squeeze(1)
            
            optimizer.zero_grad()
            
            if isinstance(model, UNet):
                outputs = model(images)
            else:  # DeepLabV3
                outputs = model(images)['out']
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            avg_loss += loss.detach().cpu()
        
        avg_loss = avg_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
        
        # Save model checkpoint periodically
        if save_path and (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"{save_path}_epoch{epoch+1}.pth")
    
    # Save final model
    if save_path:
        torch.save(model.state_dict(), f"{save_path}_final.pth")
    
    return model


# Uncertainty quantification with Monte Carlo Dropout
def mc_dropout_one_batch(model, x, num_samples, device='cuda'):
    """
    Perform Monte Carlo Dropout for a single batch.
    
    Args:
        model: The model to use
        x: Input data
        num_samples (int): Number of Monte Carlo samples
        device (str): Device to use
        
    Returns:
        list: List of model predictions
    """
    torch.cuda.empty_cache()
    
    with torch.no_grad():
        # Enable dropout during inference
        model.train()
        
        # For each module in the model, if it's a dropout layer, set train mode
        for module in model.modules():
            if isinstance(module, nn.Dropout) or isinstance(module, nn.Dropout2d):
                module.train()
                
        # Move input to device
        x = x.to(device)
        
        # Generate samples
        preds = []
        for _ in range(num_samples):
            if isinstance(model, UNet):
                pred = torch.softmax(model(x), dim=1).cpu().detach()
            else:  # DeepLabV3
                pred = torch.softmax(model(x)['out'], dim=1).cpu().detach()
            preds.append(pred)
    
    return preds


def mc_dropout_all_batches(dataloader, model, num_samples, device='cuda'):
    """
    Perform Monte Carlo Dropout for all batches in a dataloader.
    
    Args:
        dataloader: DataLoader with the data
        model: The model to use
        num_samples (int): Number of Monte Carlo samples
        device (str): Device to use
        
    Returns:
        list: List of predictions for all batches
    """
    all_predictions = []
    
    for imgs, targets in tqdm(dataloader, desc="Generating MC samples"):
        imgs = imgs.to(device).float()
        if isinstance(targets, torch.Tensor):
            targets = targets.to(device)
            if targets.dim() > 1:
                targets = targets.squeeze(1)
        
        # Get predictions with MC dropout
        pred_sets = mc_dropout_one_batch(model, imgs, num_samples, device)
        
        # Stack predictions
        img_predictions = torch.stack(pred_sets, dim=1)
        
        # Add to list of all predictions
        all_predictions.append(img_predictions)
    
    # Clean up to reduce memory usage
    torch.cuda.empty_cache()
    
    return all_predictions


def calculate_uncertainty(mc_predictions):
    """
    Calculate uncertainty metrics from Monte Carlo predictions.
    
    Args:
        mc_predictions (torch.Tensor): Predictions from MC dropout
            Shape: [batch_size, num_samples, num_classes, height, width]
            
    Returns:
        dict: Dictionary with uncertainty metrics
    """
    # Calculate standard deviation across MC samples for each class
    std_img = mc_predictions.std(dim=1)
    
    # Sum standard deviation spatially for each class
    sum_std = std_img.sum(dim=(2, 3))
    
    return {
        'std_per_pixel': std_img,
        'sum_std': sum_std
    }


def scale_to_minus_one_one(x, min_value, max_value):
    """
    Scale values to the range [-1, 1].
    
    Args:
        x: Value to scale
        min_value: Minimum value in the range
        max_value: Maximum value in the range
        
    Returns:
        float: Scaled value
    """
    return -1 + 2 * (x - min_value) / (max_value - min_value)


def quantify_class_uncertainty(num_imgs, num_classes, class_names, sum_variance_values):
    """
    Quantify uncertainty for each land cover class.
    
    Args:
        num_imgs (int): Number of images
        num_classes (int): Number of classes
        class_names (dict): Dictionary mapping class IDs to names
        sum_variance_values (torch.Tensor): Sum of variance values
        
    Returns:
        dict: Dictionary with class uncertainty values
    """
    # Initialize array for normalized standard deviation values
    class_pixel_mean_std = np.empty((num_imgs, num_classes))
    
    # Initialize dictionary for results
    classes_dictionary = {}
    
    # Scale standard deviation values to [-1, 1] range for each image
    for i in range(num_imgs):
        for j in range(num_classes):
            min_val = min(sum_variance_values[i, :])
            max_val = max(sum_variance_values[i, :])
            class_pixel_mean_std[i, j] = scale_to_minus_one_one(
                sum_variance_values[i, j], min_val, max_val
            )
    
    # Create dictionary with class names
    for i, key in enumerate(class_names.keys()):
        classes_dictionary[key] = [class_names.get(key)]
    
    # Add class index and mean standard deviation
    for i, key in enumerate(classes_dictionary.keys()):
        if len(classes_dictionary[key]) == 1:
            classes_dictionary[key].append([i, class_pixel_mean_std[:, i].mean()])
    
    return classes_dictionary


# Surrogate Model for XAI
def extract_features_from_segmentation(dataloader, model, device='cuda'):
    """
    Extract land cover class distribution features from segmentation predictions.
    
    Args:
        dataloader: DataLoader with the data
        model: Trained segmentation model
        device (str): Device to use
        
    Returns:
        tuple: (feature_array, target_labels)
    """
    # Get list of class indices
    land_cover_indices = list(range(len(CORINE_CLASSES)))
    
    feature_array_list = []
    target_labels_list = []
    
    # Set model to evaluation mode
    model.eval()
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting features"):
            # Move images to device
            images = images.to(device).float()
            
            # Get model predictions
            if isinstance(model, UNet):
                outputs = model(images)
            else:  # DeepLabV3
                outputs = model(images)['out']
            
            # Convert to segmentation masks (class with highest probability)
            segmentation_masks = outputs.argmax(dim=1)
            
            # Process each mask in the batch
            for mask in segmentation_masks:
                # Count pixels for each class
                class_counts = {idx: 0 for idx in land_cover_indices}
                
                for class_idx in torch.unique(mask):
                    pixel_count = torch.sum(mask == class_idx).item()
                    class_counts[class_idx.item()] = pixel_count
                
                # Add counts to feature list
                feature_array_list.append(list(class_counts.values()))
            
            # Add labels to target list
            if isinstance(labels, torch.Tensor) and labels.dim() > 0:
                target_labels_list.append(labels.cpu().numpy())
            else:
                # For single labels
                target_labels_list.extend(labels)
    
    # Convert to numpy arrays
    feature_array = np.array(feature_array_list)
    
    # Handle different label formats
    if isinstance(target_labels_list[0], np.ndarray):
        target_labels = np.concatenate(target_labels_list)
    else:
        target_labels = np.array(target_labels_list)
    
    return feature_array, target_labels


def train_surrogate_model(feature_array, target_labels):
    """
    Train a logistic regression surrogate model for XAI.
    
    Args:
        feature_array (numpy.ndarray): Feature array from segmentation
        target_labels (numpy.ndarray): Target labels
        
    Returns:
        tuple: (model, scaler)
    """
    # Reshape if needed
    if feature_array.ndim > 2:
        feature_array = feature_array.reshape(-1, feature_array.shape[-1])
    
    target_labels = target_labels.reshape(-1)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_array)
    
    # Train logistic regression model
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_scaled, target_labels)
    
    return logreg, scaler


def analyze_surrogate_model(logreg, class_names):
    """
    Analyze the surrogate model to identify important features.
    
    Args:
        logreg: Trained logistic regression model
        class_names (dict): Dictionary mapping class IDs to names
        
    Returns:
        tuple: (top_pos_features, top_pos_coeffs, top_neg_features, top_neg_coeffs)
    """
    # Get model coefficients
    coefficients = logreg.coef_[0]
    
    # Get feature names
    feature_names = list(class_names.values())
    
    # Sort coefficients
    sorted_indices = np.argsort(coefficients)[::-1]
    sorted_coeffs = coefficients[sorted_indices]
    sorted_features = [feature_names[i] for i in sorted_indices]
    
    # Get top 3 positive and negative features
    top_pos_features = sorted_features[:3]
    top_pos_coeffs = sorted_coeffs[:3]
    
    top_neg_features = sorted_features[-3:]
    top_neg_coeffs = sorted_coeffs[-3:]
    
    return top_pos_features, top_pos_coeffs, top_neg_features, top_neg_coeffs


def create_dashboard(top_pos_features, top_pos_coeffs, top_neg_features, top_neg_coeffs):
    """
    Create a visualization dashboard for XAI results.
    
    Args:
        top_pos_features (list): Top positive features
        top_pos_coeffs (list): Coefficients for top positive features
        top_neg_features (list): Top negative features
        top_neg_coeffs (list): Coefficients for top negative features
        
    Returns:
        tuple: (fig_pos, fig_neg) Plotly figures
    """
    # Create positive features figure
    fig_pos = go.Figure(data=[
        go.Bar(
            x=top_pos_coeffs, 
            y=top_pos_features, 
            orientation='h',
            marker=dict(color=top_pos_coeffs, colorscale='Viridis')
        )
    ])
    
    fig_pos.update_layout(
        title="Top 3 Features Contributing to Wilderness",
        xaxis_title="Coefficient Value",
        yaxis_title="Feature"
    )
    
    # Create negative features figure
    fig_neg = go.Figure(data=[
        go.Bar(
            x=top_neg_coeffs, 
            y=top_neg_features, 
            orientation='h',
            marker=dict(color=top_neg_coeffs, colorscale='Viridis')
        )
    ])
    
    fig_neg.update_layout(
        title="Top 3 Features Contributing to Non-Wilderness",
        xaxis_title="Coefficient Value",
        yaxis_title="Feature"
    )
    
    return fig_pos, fig_neg


# Main workflow functions
def train_segmentation_model(config):
    """
    Train a semantic segmentation model.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        model: Trained model
    """
    # Setup data
    transform, transform_target = create_transformations()
    
    dataset = CustomDataset(
        csv_file=config['csv_file'],
        image_folder=config['image_folder'],
        target_folder=config['target_folder'],
        transform_image=transform,
        transform_target=transform_target
    )
    
    train_dataloader, test_dataloader = create_dataloaders(
        dataset, 
        train_ratio=config['train_ratio'],
        batch_size=config['batch_size']
    )
    
    # Setup model
    model = setup_model(
        model_type=config['model_type'],
        input_channels=config['input_channels'],
        output_classes=config['output_classes'],
        device=config['device']
    )
    
    # Train model
    trained_model = train_model(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        num_epochs=config['num_epochs'],
        learning_rate=config['learning_rate'],
        device=config['device'],
        save_path=config['save_path']
    )
    
    return trained_model


def quantify_uncertainty(config):
    """
    Quantify uncertainty using Monte Carlo dropout.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        dict: Uncertainty results
    """
    # Setup data
    transform, transform_target = create_transformations()
    
    dataset = CustomDataset(
        csv_file=config['csv_file'],
        image_folder=config['image_folder'],
        target_folder=config['target_folder'],
        transform_image=transform,
        transform_target=transform_target
    )
    
    _, test_dataloader = create_dataloaders(
        dataset, 
        train_ratio=config['train_ratio'],
        batch_size=config['batch_size']
    )
    
    # Load model
    model = setup_model(
        model_type=config['model_type'],
        input_channels=config['input_channels'],
        output_classes=config['output_classes'],
        device=config['device'],
        checkpoint_path=config['checkpoint_path']
    )
    
    # Generate MC dropout samples
    mc_predictions = mc_dropout_all_batches(
        dataloader=test_dataloader,
        model=model,
        num_samples=config['num_samples'],
        device=config['device']
    )
    
    # Convert to tensor
    mc_predictions_tensor = torch.cat(mc_predictions, dim=0)
    
    # Calculate uncertainty
    uncertainty = calculate_uncertainty(mc_predictions_tensor)
    
    # Quantify class uncertainty
    num_imgs = uncertainty['sum_std'].size(0)
    num_classes = uncertainty['sum_std'].size(1)
    
    class_uncertainty = quantify_class_uncertainty(
        num_imgs=num_imgs,
        num_classes=num_classes,
        class_names=CORINE_CLASSES,
        sum_variance_values=uncertainty['sum_std']
    )
    
    return {
        'mc_predictions': mc_predictions_tensor,
        'uncertainty': uncertainty,
        'class_uncertainty': class_uncertainty
    }


def build_surrogate_model(config):
    """
    Build and analyze a surrogate model for XAI.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        dict: Surrogate model results
    """
    # Setup data
    transform, _ = create_transformations()
    
    dataset = CustomDataset(
        csv_file=config['csv_file'],
        image_folder=config['image_folder'],
        transform_image=transform
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        drop_last=False
    )
    
    # Load model
    model = setup_model(
        model_type=config['model_type'],
        input_channels=config['input_channels'],
        output_classes=config['output_classes'],
        device=config['device'],
        checkpoint_path=config['checkpoint_path']
    )
    
    # Extract features from segmentation predictions
    feature_array, target_labels = extract_features_from_segmentation(
        dataloader=dataloader,
        model=model,
        device=config['device']
    )
    
    # Train surrogate model
    logreg, scaler = train_surrogate_model(
        feature_array=feature_array,
        target_labels=target_labels
    )
    
    # Analyze surrogate model
    top_pos_features, top_pos_coeffs, top_neg_features, top_neg_coeffs = analyze_surrogate_model(
        logreg=logreg,
        class_names=CORINE_CLASSES
    )
    
    # Create visualization dashboard
    fig_pos, fig_neg = create_dashboard(
        top_pos_features=top_pos_features,
        top_pos_coeffs=top_pos_coeffs,
        top_neg_features=top_neg_features,
        top_neg_coeffs=top_neg_coeffs
    )
    
    return {
        'logreg': logreg,
        'scaler': scaler,
        'top_positive_features': top_pos_features,
        'top_positive_coefficients': top_pos_coeffs,
        'top_negative_features': top_neg_features,
        'top_negative_coefficients': top_neg_coeffs,
        'visualizations': {
            'positive': fig_pos,
            'negative': fig_neg
        }
    }


def end_to_end_pipeline(config):
    """
    Run the complete naturalness assessment pipeline.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        dict: Results from all pipeline components
    """
    results = {}
    
    # Step 1: Train segmentation model (if needed)
    if config.get('train_model', False):
        print("Training segmentation model...")
        model = train_segmentation_model(config)
        results['trained_model'] = model
    
    # Step 2: Quantify uncertainty with MC dropout
    print("Quantifying uncertainty...")
    uncertainty_results = quantify_uncertainty(config)
    results['uncertainty'] = uncertainty_results
    
    # Step 3: Build surrogate model for XAI
    print("Building surrogate model for XAI...")
    surrogate_results = build_surrogate_model(config)
    results['surrogate'] = surrogate_results
    
    # Save results if path is provided
    if 'results_path' in config:
        # Save class uncertainty dictionary
        np.save(
            f"{config['results_path']}_class_uncertainty.npy", 
            uncertainty_results['class_uncertainty']
        )
        
        # Save feature array and target labels for surrogate model
        np.save(
            f"{config['results_path']}_feature_array.npy",
            surrogate_results.get('feature_array')
        )
        np.save(
            f"{config['results_path']}_target_labels.npy",
            surrogate_results.get('target_labels')
        )
        
        # Save visualization figures
        if 'visualizations' in surrogate_results:
            pos_fig = surrogate_results['visualizations']['positive']
            neg_fig = surrogate_results['visualizations']['negative']
            
            pos_fig.write_html(f"{config['results_path']}_positive_features.html")
            neg_fig.write_html(f"{config['results_path']}_negative_features.html")
    
    return results


# Example usage
if __name__ == "__main__":
    # Configuration dictionary
    config = {
        # Data settings
        'csv_file': 'infos.csv',
        'image_folder': 'data/anthroprotect/tiles/s2',
        'target_folder': 'data/anthroprotect/new_masks',
        'train_ratio': 0.8,
        'batch_size': 4,
        
        # Model settings
        'model_type': 'deeplabv3',  # 'deeplabv3' or 'unet'
        'input_channels': 3,  # 3 for DeepLabV3, 10 for UNet with all channels
        'output_classes': 43,  # Number of CORINE land cover classes
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'checkpoint_path': 'checkpoint/model_final.pth',  # Path to saved model
        
        # Training settings
        'train_model': False,  # Set to True to train the model
        'num_epochs': 100,
        'learning_rate': 0.001,
        'save_path': 'checkpoint/model',
        
        # Uncertainty settings
        'num_samples': 25,  # Number of Monte Carlo samples
        
        # Results settings
        'results_path': 'results/naturalness_assessment',
    }
    
    # Run the pipeline
    results = end_to_end_pipeline(config)
    
    print("Pipeline completed successfully!")
    
    # Print key results
    print("\nUncertainty Results:")
    print("--------------------")
    for class_id, (name, data) in results['uncertainty']['class_uncertainty'].items():
        if isinstance(data, list) and len(data) > 1:
            print(f"Class {class_id} ({name}): Uncertainty = {data[1]:.4f}")
    
    print("\nXAI Results:")
    print("------------")
    print("Top 3 features contributing to wilderness:")
    for feature, coef in zip(
        results['surrogate']['top_positive_features'], 
        results['surrogate']['top_positive_coefficients']
    ):
        print(f"- {feature}: {coef:.4f}")
    
    print("\nTop 3 features contributing to non-wilderness:")
    for feature, coef in zip(
        results['surrogate']['top_negative_features'], 
        results['surrogate']['top_negative_coefficients']
    ):
        print(f"- {feature}: {coef:.4f}")
