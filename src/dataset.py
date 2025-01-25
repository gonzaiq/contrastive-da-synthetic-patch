import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import numpy as np
import json

# Define the LUT function
def inv_lut_fn(image, image_max=255, w_width=40, w_center=210, epsilon=1e-12):

    # Convert image to numpy array and normalize to 0-1 range
    np_image = np.array(image, dtype=np.float32)
    
    # Apply the formula:
    transformed_image = -w_width * np.log((image_max - np_image) / (np_image + epsilon) + epsilon ) / 4 + w_center
    
    # Normalize to the range 0-255 and convert to uint8
    transformed_image = np.clip(transformed_image, 0, 255)
    transformed_image = np.array(transformed_image, dtype=np.uint8)
    
    # Return the processed image
    return Image.fromarray(transformed_image)

class CustomDataset(Dataset):
    def __init__(self, data_list, transform=None):
        """
        Args:
            data_list (list): A list of dictionaries containing 'image', 'breast_mask', 'label', 'lut_flag'
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = self.data_list[idx]
        
        # Load images
        image = Image.open(sample['image']).convert('L')
        breast_mask = Image.open(sample['breast_mask']).convert('L')  # Assuming mask is grayscale
        
        label = sample['label']
        lut_flag = sample['lut_flag']

        # Apply LUT function if lut_flag is True
        if not lut_flag:
            image = inv_lut_fn(image)

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)
            breast_mask = self.transform(breast_mask)

        return image, breast_mask, label, lut_flag

# Load the dataset
def load_dataset(datalist_pth):
    with open(datalist_pth, 'r') as file:
        datalist = json.load(file)
    return datalist

def masked_normalization(img, mask, ce_gamma=None):

    if (ce_gamma is not None) and (ce_gamma < 1.0):

        # classical masked normalization
        _mean = 0.5*img[mask == 1].mean() + 0.5*img[mask == 0].mean() 
        _std = np.sqrt(np.power(img - _mean, 2).mean())

        return (img - _mean) / _std
    else:
        _mean = 0.5 * img[mask == 0].max() + 0.5 * img[mask == 1].min()
        _std = np.sqrt(np.power(img[mask == 1] - _mean, 2).mean())

        return (img - _mean) / _std - 1

def masked_normalization_universal(img, mask, ce_gamma=None):

    _mean_breast = img[mask == 1].mean()
    _std_breast = img[mask == 1].std()

    # normalize
    img_norm = (img - _mean_breast) / _std_breast

    # background set to the minimal value of the breast
    _min_breast = img_norm[mask == 1].min()
    img_norm[mask == 0] = _min_breast

    return img_norm

def apply_inverse_lut(img, ww, wc, out_range=3500, epsilon=10**-12):

    inv_lut = lambda x: -ww/4 * (np.log(out_range - x + epsilon) - np.log(x + epsilon)) + wc
    img_transf = inv_lut(img)
    img_transf[img_transf < 0] = 0 
    img_transf[img_transf > 255] = 255 

    return img_transf
