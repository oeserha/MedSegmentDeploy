# Dataloader & Box Integration

import pandas as pd
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split
import os
import boto3
import io

#TODO: edit path
PATH = "/home/ra-ugrad/Documents/Segmentations/"

class CancerDataset(Dataset):
    def __init__(self, labels, path, train=True, transform=None, target_transform=None):
        self.img_labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.path = path

        self.value_to_class = {
            9362: 0,   # Background
            18724: 1,  # Water
            28086: 2,  # Skin
            37449: 3,  # Fat
            46811: 4,  # FGT
            56174: 5,  # Tumor
            65536: 6   # Clip
        }
    
    def __len__(self):
        return int(sum(self.img_labels['Total Images']))
    
    def __getitem__(self, idx):
        # Get row based on index
        start_idx = self.img_labels.Start_Index
        input_test = self.img_labels.Start_Index.where(start_idx >= idx).first_valid_index()
        if input_test is None:
            input_test = self.img_labels.Start_Index.last_valid_index()
        row = self.img_labels.loc[input_test]
        
        # Calculate brightness level and image index
        patient = row['patient_id']
        bright_level = int((idx - row['Start_Index']) % row['Number of Brightness Levels'])
        if bright_level == 0:
            bright_id = f"Brightness level {row["Brightness Level 1"]}"
        else:
            bright_id = f"Brightness level {row["Brightness Level 2"]}"

        img_idx = int(((idx - row['Start_Index']) % row['Number of Slices'])+1)
        img_idx = f"{img_idx:05d}"

        # S3 details
        bucket = self.path
        key = f'MRIs/{patient}/MRI PNGs/{bright_id}/png_{img_idx}.png'
        print(img_idx)
        
        # Download object from S3
        s3 = boto3.client('s3')
        response = s3.get_object(Bucket=bucket, Key=key)
        image_data = response['Body'].read()
        
        try:
            image = Image.open(io.BytesIO(image_data))
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading image: {key}")
            print(f"Error: {str(e)}")
            return None

        # Load and process segmentation mask
        try:
            seg_key = f'MRIs/{patient}/Seg PNGs/segpng_{img_idx}.png'
            seg_response = s3.get_object(Bucket=bucket, Key=seg_key)
            seg_image_data = seg_response['Body'].read()
            label = np.array(Image.open(io.BytesIO(seg_image_data)))
            
            label_classes = torch.zeros(label.shape, dtype=torch.long)
            
            for value, class_idx in self.value_to_class.items():
                label_classes[label == value] = class_idx
            
            max_class = label_classes.max().item()
            if max_class >= 7:
                print(f"Warning: Invalid class index found: {max_class}")
                return None

            # the resnet I use needs (224,224) but feel free to change it.
            label_classes = label_classes.unsqueeze(0).unsqueeze(0)  
            label_classes = F.interpolate(
                label_classes.float(),
                size=(224, 224),
                mode='nearest'
            )
            label_classes = label_classes.squeeze(0).squeeze(0).long() 
            
        except Exception as e:
            print(f"Error loading mask: {self.path}{patient}/Seg PNGs/segpng_{img_idx}.png")
            print(f"Error: {str(e)}")
            return None 
        
        # convert to tensor
        image = torch.Tensor(np.array(image))
        label_classes = torch.Tensor(np.array(label_classes))

        return image, label_classes, patient, bright_id 

# Additional Fields for MRI Data
def clean_mri_data(mri_data):
    mri_data['Total Images'] = mri_data['Number of Slices'] * mri_data['Number of Brightness Levels']
    mri_data['Start_Index'] = 0

    mri_data.rename(columns={"MRI/Patient ID": "patient_id"}, inplace=True)
    mri_data = mri_data.reset_index()
    mri_dummy = "MRI PNGs"

    for i in range(len(mri_data)):
        mri_data.loc[i, 'Start_Index'] = sum(mri_data['Total Images'][:i])

    return mri_data

# def train_test(mri_data, split=0.25):
#     # Train/Test split
#     train_data, test_data = train_test_split(mri_data, test_size=split)
#     train_data = train_data.reset_index().drop(columns = 'index')
#     test_data = test_data.reset_index().drop(columns = 'index')

#     for i in range(len(train_data)):
#         train_data.loc[i, 'Start_Index'] = sum(train_data['Total Images'][:i])
        
#     for i in range(len(test_data)):
#         test_data.loc[i, 'Start_Index'] = sum(test_data['Total Images'][:i])

#     return train_data, test_data

def calculate_iou(pred, target, num_classes):
    # Convert predictions to class indices
    pred = torch.argmax(pred, dim=1)
    
    # Initialize IoU for each class
    ious = []
    
    # Calculate IoU for each class
    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        
        intersection = list((pred_inds & target_inds).sum(dim=(1,2)).float().cpu().numpy())
        union = list((pred_inds | target_inds).sum(dim=(1,2)).float().cpu().numpy())

        #TODO: change this so 0s are 100s

        iou = [x / (y + 1e-6) for x, y in zip(intersection, union)]
        ious.append(iou)

    first_values = [sublist[0] for sublist in ious]
    second_values = [sublist[1] for sublist in ious]
    first_mean = sum(first_values) / len(ious)
    second_mean = sum(second_values) / len(ious)
    
    return [first_mean, second_mean], ious
