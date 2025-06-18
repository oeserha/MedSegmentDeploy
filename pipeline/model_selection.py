import json
import sys
import time
import boto3
import os
import subprocess
import openpyxl
import torch
import pandas as pd
import numpy as np

os.chdir("..")

from smart_open import open as smart_open
from torch.utils.data import DataLoader
import io
from segment_anything import sam_model_registry
from torch.nn import Linear
from torch.nn import Embedding
from segment_anything.modeling.mask_decoder import MLP
import torch.nn as nn
from segmentation_models_pytorch import Unet
from src.dino_helper import DINOv2Segmentation
from torchvision import transforms
from src.data_helper import calculate_iou
import src.data_helper as data_helper
import src.medsam_helper as medsam_helper
import src.credentials as credentials

# get data
s3 = boto3.client('s3', aws_access_key_id=credentials.ACCESS_KEY, aws_secret_access_key=credentials.SECRET_KEY)

response = s3.get_object(Bucket='raw-data-mris-segs', Key='seg_list_test.xlsx') 
data = response['Body'].read()
patient_data = pd.read_excel(io.BytesIO(data))

cleaned_data = data_helper.clean_mri_data(patient_data)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
test_dataset = data_helper.CancerDataset(labels=cleaned_data, path='cleaned-mri-data', train=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

# load models
model_bucket = "medseg-models"
base_path = "base_medsam_model_06-10-2025.pth"
tuned_path = "tuned_medsam_model_06-10-2025.pth"
unet_path = "unet_model_06-11-2025.pth"
dino_path = "dinov2_model_06-11-2025.pth"
medsam_path = "/home/ra-ugrad/Documents/Haleigh/MedicalImage/models/medsam_vit_b.pth"

base_model = sam_model_registry['vit_b'](checkpoint=medsam_path)
base_model.mask_decoder.num_mask_tokens = 8
base_model.mask_decoder.num_multimask_outputs = 7
base_model.image_encoder.patch_embed.proj = nn.Conv2d(3, 768, kernel_size = (35, 35), stride = (3, 3))
base_model.mask_decoder.mask_tokens = Embedding(base_model.mask_decoder.num_mask_tokens, 256)
base_model.mask_decoder.output_hypernetworks_mlps = nn.ModuleList([MLP(256, 256, 32, 3) for i in range(base_model.mask_decoder.num_mask_tokens)])
base_model.mask_decoder.iou_prediction_head.layers[2] = Linear(in_features=256, out_features=base_model.mask_decoder.num_mask_tokens, bias=True)

tuned_model = sam_model_registry['vit_b'](checkpoint=medsam_path)
tuned_model.mask_decoder.num_mask_tokens = 8
tuned_model.mask_decoder.num_multimask_outputs = 7
tuned_model.image_encoder.patch_embed.proj = nn.Conv2d(3, 768, kernel_size = (35, 35), stride = (3, 3))
tuned_model.mask_decoder.mask_tokens = Embedding(tuned_model.mask_decoder.num_mask_tokens, 256)
tuned_model.mask_decoder.output_hypernetworks_mlps = nn.ModuleList([MLP(256, 256, 32, 3) for i in range(tuned_model.mask_decoder.num_mask_tokens)])
tuned_model.mask_decoder.iou_prediction_head.layers[2] = Linear(in_features=256, out_features=tuned_model.mask_decoder.num_mask_tokens, bias=True)

unet_model = Unet(encoder_name='resnet34', encoder_weights='imagenet', in_channels=1, classes=7)
dino_model = DINOv2Segmentation()

base_response = s3.get_object(Bucket=model_bucket, Key=base_path) 
base_data = base_response['Body'].read()
base_model.load_state_dict(torch.load(io.BytesIO(base_data)))

tuned_response = s3.get_object(Bucket=model_bucket, Key=tuned_path) 
tuned_data = tuned_response['Body'].read()
tuned_model.load_state_dict(torch.load(io.BytesIO(tuned_data)))

unet_response = s3.get_object(Bucket=model_bucket, Key=unet_path) 
unet_data = unet_response['Body'].read()
unet_model.load_state_dict(torch.load(io.BytesIO(unet_data)))

dino_response = s3.get_object(Bucket=model_bucket, Key=dino_path) 
dino_data = dino_response['Body'].read()
dino_model.load_state_dict(torch.load(io.BytesIO(dino_data)))

# get results from each model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# medsam

tuned_results = None 
tuned_model = tuned_model.to(device)
for img, seg, patient, b_level in test_loader:
    img = img.to(device)
    seg = seg.to(device)
    # print(img.size())
    B, C, H, W = img.size()
    img_3c = img

    box_np = torch.Tensor(np.array([[0, 0, W, H]])).to(device)

    with torch.no_grad():
        image_embedding = tuned_model.image_encoder(img_3c) 
    
    medsam_seg = medsam_helper.medsam_inference(tuned_model, image_embedding, box_np, H, W)
    pred = torch.argmax(medsam_seg, dim=1).to(device)

    acc = list((pred == seg).float().mean(dim =(1, 2)).cpu().numpy()) #TODO: would be helpful to see acc by mask
    mean_iou, class_iou = calculate_iou(medsam_seg.cpu(), seg.cpu(), 7)

    if tuned_results is None:
        tuned_results = pd.DataFrame({"Patient": patient, 
                                        "Brightness": b_level, 
                                        "Accuracy": acc,
                                        "IoU_0": class_iou[0],
                                        "IoU_1": class_iou[1],
                                        "IoU_2": class_iou[2],
                                        "IoU_3": class_iou[3], 
                                        "IoU_4": class_iou[4],
                                        "IoU_5": class_iou[5],
                                        "IoU_6": class_iou[6],
                                        "IoU_mean": mean_iou,
                                        })
    else:
        tuned_results = pd.concat([tuned_results, pd.DataFrame({"Patient": patient, 
                                                                "Brightness": b_level, 
                                                                "Accuracy": acc,
                                                                "IoU_0": class_iou[0],
                                                                "IoU_1": class_iou[1],
                                                                "IoU_2": class_iou[2],
                                                                "IoU_3": class_iou[3], 
                                                                "IoU_4": class_iou[4],
                                                                "IoU_5": class_iou[5],
                                                                "IoU_6": class_iou[6],
                                                                "IoU_mean": mean_iou,
                                                                })]) 
grouped_tuned_results = tuned_results.groupby(["Patient", "Brightness"]).mean()
grouped_tuned_results['Model'] = 'MedSAM'

# unet
unet_results = None
unet_model.eval()
unet_model = unet_model.to(device)
with torch.no_grad():
    for img, seg, patient, b_level in test_loader:
        img = img[:,:1,:,:] # gets just one channel, since img is black and white
        img = img.to(device)
        seg = seg.to(device)
        outputs = unet_model(img)
        pred = torch.argmax(outputs, dim=1)

        acc = list((pred == seg).float().mean(dim =(1, 2)).cpu().numpy())
        mean_iou, class_iou = calculate_iou(outputs.cpu(), seg.cpu(), 7)

        if unet_results is None:
            unet_results = pd.DataFrame({"Patient": patient, 
                                            "Brightness": b_level, 
                                            "Accuracy": acc,
                                            "IoU_0": class_iou[0],
                                            "IoU_1": class_iou[1],
                                            "IoU_2": class_iou[2],
                                            "IoU_3": class_iou[3], 
                                            "IoU_4": class_iou[4],
                                            "IoU_5": class_iou[5],
                                            "IoU_6": class_iou[6],
                                            "IoU_mean": mean_iou,
                                            })
        else:
            unet_results = pd.concat([unet_results, pd.DataFrame({"Patient": patient, 
                                                                    "Brightness": b_level, 
                                                                    "Accuracy": acc,
                                                                    "IoU_0": class_iou[0],
                                                                    "IoU_1": class_iou[1],
                                                                    "IoU_2": class_iou[2],
                                                                    "IoU_3": class_iou[3], 
                                                                    "IoU_4": class_iou[4],
                                                                    "IoU_5": class_iou[5],
                                                                    "IoU_6": class_iou[6],
                                                                    "IoU_mean": mean_iou,
                                                                    })]) 

grouped_unet_results = unet_results.groupby(["Patient", "Brightness"]).mean()
grouped_unet_results['Model'] = 'Unet'

# dino
dino_model = dino_model.to(device)
dino_model.eval()
dino_results = None
with torch.no_grad():
    for img, seg, patient, b_level in test_loader:
        img = img.to(device)
        seg = seg.to(device)

        B, C, H, W = img.size()
        img_3c = img
        
        outputs = dino_model(img_3c)
        outputs = outputs.view(-1, 7, 224, 224)
        pred = torch.argmax(outputs, dim=1)
        
        acc = list((pred == seg).float().mean(dim =(1, 2)).cpu().numpy())
        mean_iou, class_iou = calculate_iou(outputs, seg, 7)

        if dino_results is None:
            dino_results = pd.DataFrame({"Patient": patient, 
                                            "Brightness": b_level, 
                                            "Accuracy": acc,
                                            "IoU_0": class_iou[0],
                                            "IoU_1": class_iou[1],
                                            "IoU_2": class_iou[2],
                                            "IoU_3": class_iou[3], 
                                            "IoU_4": class_iou[4],
                                            "IoU_5": class_iou[5],
                                            "IoU_6": class_iou[6],
                                            "IoU_mean": mean_iou,
                                            })
        else:
            dino_results = pd.concat([dino_results, pd.DataFrame({"Patient": patient, 
                                                                    "Brightness": b_level, 
                                                                    "Accuracy": acc,
                                                                    "IoU_0": class_iou[0],
                                                                    "IoU_1": class_iou[1],
                                                                    "IoU_2": class_iou[2],
                                                                    "IoU_3": class_iou[3], 
                                                                    "IoU_4": class_iou[4],
                                                                    "IoU_5": class_iou[5],
                                                                    "IoU_6": class_iou[6],
                                                                    "IoU_mean": mean_iou,
                                                                    })]) 

grouped_dino_results = dino_results.groupby(["Patient", "Brightness"]).mean()
grouped_dino_results['Model'] = 'DINO'

all_results = pd.concat([grouped_tuned_results, grouped_unet_results, grouped_dino_results])
avg_results = all_results[['Accuracy', 'IoU_0', 'IoU_1', 'IoU_2', 'IoU_3', 'IoU_4', 'IoU_5', 'IoU_6', 'IoU_mean', 'Model']].groupby('Model').mean()
dino_sum = avg_results.loc['dino'].sum()
unet_sum = avg_results.loc['unet'].sum()
medsam_sum = avg_results.loc['medsam'].sum()

model = max([medsam_sum, unet_sum, dino_sum])