import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn as nn

from src.utils import calculate_iou

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='blue', facecolor=(0,0,0,0), lw=2))

def medsam_inference(medsam_model, img_embed, box_1024, H, W):
    with torch.no_grad():
        box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
        if len(box_torch.shape) == 2:
            box_torch = box_torch[:, None, :] # (B, 1, 4)

        sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
            points=None,
            boxes=box_torch,
            masks=None,
        )
        low_res_logits, _ = medsam_model.mask_decoder(
            image_embeddings=img_embed, # (B, 256, 64, 64)
            image_pe=medsam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=True,
            )

        low_res_pred = torch.softmax(low_res_logits, dim=1)  # (1, 1, 256, 256)

        low_res_pred = F.interpolate(
            low_res_pred,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )  # (1, 1, gt.shape)
        low_res_pred = low_res_pred.squeeze().cpu()
        return low_res_pred
    
def medsam_test_results(test_loader, medsam_model, device):
    results = None 
    medsam_model = medsam_model.to(device)
    for img, seg, patient, b_level in test_loader:
        img = img.to(device)
        seg = seg.to(device)
        # print(img.size())
        B, C, H, W = img.size()
        img_3c = img.repeat(3, 1, 1, 1).view(B, 3, H, W).to(device)

        box_np = torch.Tensor(np.array([[0, 0, W, H]])).to(device)

        with torch.no_grad():
            image_embedding = medsam_model.image_encoder(img_3c) 
        
        medsam_seg = medsam_inference(medsam_model, image_embedding, box_np, H, W)
        pred = torch.argmax(medsam_seg, dim=1).to(device)

        acc = list((pred == seg).float().mean(dim =(1, 2)).cpu().numpy()) #TODO: would be helpful to see acc by mask
        mean_iou, class_iou = calculate_iou(medsam_seg.cpu(), seg.cpu(), 7)

        if results is None:
            results = pd.DataFrame({"Patient": patient, 
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
            results = pd.concat([results, pd.DataFrame({"Patient": patient, 
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

    return results

class MedSAM(nn.Module):
    def __init__(
        self,
        image_encoder,
        mask_decoder,
        prompt_encoder,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder

        # freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image, box):
        B, C, H, W = image.size()
        image_embedding = self.image_encoder(image)  

        # do not compute gradients for prompt encoder
        with torch.no_grad():
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]  

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,  
            image_pe=self.prompt_encoder.get_dense_pe(),  
            sparse_prompt_embeddings=sparse_embeddings,  
            dense_prompt_embeddings=dense_embeddings, 
            multimask_output=True,
        )

        low_res_pred = torch.softmax(low_res_masks, dim=1) 

        low_res_pred = F.interpolate(
            low_res_pred,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )  
        
        return low_res_pred