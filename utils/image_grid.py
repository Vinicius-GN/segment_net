
import os
import cv2

import numpy as np

import torch
from torch.utils.data import DataLoader
from utils.utils import  color_image

import matplotlib.pyplot as plt

from .segmentnet import SegmentNet

def save_grid_samples(model:SegmentNet, 
                      dataset:DataLoader, 
                      name:str, 
                      save_dir:str, 
                      mapping:dict,
                      device:torch.device):
    
    model.eval()

    #height, width
    image_size = (600, 400)

    images = []
    labels = []
    predictions =[]

    for idx, (x_img, y_img ) in enumerate(dataset):
               
        x_img = x_img.to(device)
     
        y_pred_img = model.predict(x_img, normalize=True)

        y_img = y_img.cpu().numpy()
        x_img = x_img.cpu().numpy()
        y_pred_img = y_pred_img.cpu().numpy()  

        if y_img.ndim == 4:
            y_img = [color_image(y_img[i].squeeze(), mapping) for i in range(y_img.shape[0])]
            y_img = np.asarray(y_img)

            y_pred_img = [color_image(y_pred_img[i], mapping) for i in range(y_pred_img.shape[0])]
            y_pred_img = np.asarray(y_pred_img)

            images.extend([np.transpose(x_img[i], (1,2,0)) for i in range(x_img.shape[0])])
            labels.extend([y_img[i] for i in range(y_img.shape[0])])
            predictions.extend([y_pred_img[i] for i in range(y_pred_img.shape[0])])

    rows, cols = len(images), 3
    fig, axes = plt.subplots(ncols=cols, nrows=rows,  figsize=(12, 8))
    
    for i in range(rows):

        images[i] = cv2.resize(images[i], image_size, interpolation = cv2.INTER_LINEAR).astype(np.int32)
        labels[i] = cv2.resize(labels[i], image_size, interpolation = cv2.INTER_NEAREST).astype(np.int32)
        predictions[i] = cv2.resize(predictions[i], image_size, interpolation = cv2.INTER_NEAREST).astype(np.int32)

        axes[i, 0].imshow(images[i])
        axes[i, 1].imshow(labels[i])
        axes[i, 2].imshow(predictions[i])

        axes[i, 0].axis('off')
        axes[i, 1].axis('off')
        axes[i, 2].axis('off')

    axes[0, 0].set_title("Image")
    axes[0, 1].set_title("Label")
    axes[0, 2].set_title("Prediction")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{name}.png"))
    plt.close("all")