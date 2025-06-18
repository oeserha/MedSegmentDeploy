from PIL import Image
import os
import cv2
import pandas as pd
import numpy as np

os.chdir("..")

import boto3
import src.credentials as credentials
import io

s3 = boto3.client('s3', aws_access_key_id=credentials.ACCESS_KEY, aws_secret_access_key=credentials.SECRET_KEY)

response = s3.get_object(Bucket='raw-data-mris-segs', Key='seg_list_test.xlsx') 
data = response['Body'].read()
patient_data = pd.read_excel(io.BytesIO(data))

patients = pd.read_csv("/home/ra-ugrad/Documents/Haleigh/MedSegmentDeploy/preprocessing.csv") #TODO: change to read s3
map_bool = {"Yes": True, "No": False}
patients['Reorientation'] = patients.Reorientation.map(map_bool)
patients['Renumber'] = patients.Renumber.map(map_bool)
patients = patients[patients['Patient ID'].isin(patient_data['MRI/Patient ID'])]

new_bucket = 'cleaned-mri-data'
old_bucket = 'raw-data-mris-segs'
for patient in patients['Patient ID']:
    row = patients[patients['Patient ID'] == patient]
    renum = row.Renumber.item()
    reorient = row.Reorientation.item()
    slices = patient_data[patient_data['MRI/Patient ID'] == patient]['Number of Slices'].item()
    level_1 = patient_data[patient_data['MRI/Patient ID'] == patient]['Brightness Level 1'].item()
    level_2 = patient_data[patient_data['MRI/Patient ID'] == patient]['Brightness Level 2'].item()
    levels = [level_1, level_2]
    for i in range(1, slices+1):
        img_num = str(i).zfill(5)
        for n, b in enumerate(levels):
            # get image
            old_key = f"MRIs/{patient}/MRI PNGs/Brightness level {b}/png_{img_num}.png"
            response = s3.get_object(Bucket=old_bucket, Key=old_key)
            image_data = response['Body'].read()
            img = Image.open(io.BytesIO(image_data))

            # perform operations as needed
            if reorient:
                modified = cv2.rotate(np.array(img), cv2.ROTATE_180)
                success, modified = cv2.imencode('.png', modified)
            else:
                success, modified = cv2.imencode('png', np.array(img))
            
            if renum:
                new_num = slices - (i)
                new_img_num = str(new_num).zfill(5)
                new_key = f"MRIs/{patient}/MRI PNGs/Brightness level {n+1}/png_{new_img_num}.png"
            else:
                new_key = f"MRIs/{patient}/MRI PNGs/Brightness level {n+1}/png_{img_num}.png"

            # save image to new location
            buffer = io.BytesIO(modified.tobytes())
            s3.upload_fileobj(buffer, new_bucket, new_key, ExtraArgs={'ContentType': 'image/png'})

            # save segmentation to new location
            seg_key = f"MRIs/{patient}/Seg PNGs/segpng_{img_num}.png"
            response = s3.get_object(Bucket=old_bucket, Key=seg_key)
            seg_data = response['Body'].read()
            seg = Image.open(io.BytesIO(seg_data))
            buffer = io.BytesIO()
            seg.save(buffer, format='PNG')
            buffer.seek(0)
            s3.upload_fileobj(buffer, new_bucket, seg_key)