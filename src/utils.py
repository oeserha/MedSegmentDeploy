import torch
import numpy as np

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

def get_class_weights():
    class_weights =  torch.tensor([ 1.3570,  1.6794,  7.6094,  3.8674,  7.3526, 51.9940, 99.0545])
    class_weights = class_weights/class_weights.sum()
    return torch.tensor(class_weights, dtype=torch.float32)
