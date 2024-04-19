from rembg import remove
from PIL import Image, ImageDraw
import torch
import numpy as np
import pandas as pd 
import os
import cv2
import matplotlib.pyplot as plt

# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# Convert PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)
    
# 将多个分镜合并成一张漫画
def combine_image(image_p1, image_p2, image_p3):

    # 分割斜线
    dot1_start = (0,512)
    dot1_end = (1024,400)
    dot2_start = ((int((dot1_start[0] + dot1_end[0]) / 2)), int((dot1_start[1] + dot1_end[1]) / 2))
    dot2_end = (600,1024)

    # 分割线宽度
    thickness = 10


    # 读取图片
    image1 = np.array(image_p1)

    image2_origin = np.array(image_p2)
    image2_origin = cv2.resize(image2_origin,(dot2_end[0],dot2_end[0]))
    image2 = np.zeros((1024, 1024, 3), dtype=np.uint8)
    image2[1024-dot2_end[0]:1024, 0:dot2_end[0], :] = image2_origin

    image3_origin = np.array(image_p3)
    image3_origin = cv2.resize(image3_origin,(1024-dot1_end[1],1024-dot1_end[1]))
    image3 = np.zeros((1024, 1024, 3), dtype=np.uint8)
    image3[dot1_end[1]:1024, dot1_end[1]:1024, :] = image3_origin

    # 分割
    mask_image = np.ones((1024, 1024, 3), dtype=np.uint8) * 225

    region1 = np.array([[0, 0],[1024,0], [dot1_end[0],dot1_end[1]], [dot1_start[0], dot1_start[1]]], dtype=np.int32)
    region2 = np.array([[dot1_start[0], dot1_start[1]], [dot2_start[0], dot2_start[1]], [dot2_end[0], dot2_end[1]],[0,1024]], dtype=np.int32)

    cv2.fillPoly(mask_image, [region1], (0,0,0))
    cv2.fillPoly(mask_image, [region2], (125,125,125))
    result_image =  np.ones((1024, 1024, 3), dtype=np.uint8) * 255
    mask_indices1 = np.where(mask_image == 0)
    mask_indices2 = np.where(mask_image == 125)
    mask_indices3 = np.where(mask_image == 225)
    for channel in range(3):
        result_image[mask_indices1[0], mask_indices1[1], channel] = image1[mask_indices1[0], mask_indices1[1], channel]
        result_image[mask_indices2[0], mask_indices2[1], channel] = image2[mask_indices2[0], mask_indices2[1], channel]
        result_image[mask_indices3[0], mask_indices3[1], channel] = image3[mask_indices3[0], mask_indices3[1], channel]

    # 画线
    cv2.line(result_image,(0,0),(1024,0),(0,0,0),thickness*3,cv2.LINE_AA)
    cv2.line(result_image,(0,0),(0,1024),(0,0,0),thickness*3,cv2.LINE_AA)
    cv2.line(result_image,(1024,0),(1024,1024),(0,0,0),thickness*3,cv2.LINE_AA)
    cv2.line(result_image,(0,1024),(1024,1024),(0,0,0),thickness*3,cv2.LINE_AA)
    cv2.line(result_image,dot1_start,dot1_end,(0,0,0),thickness*2,cv2.LINE_AA)    
    cv2.line(result_image,dot2_start,dot2_end,(0,0,0),thickness*2,cv2.LINE_AA) 
    cv2.line(result_image,dot1_start,dot1_end,(255,255,255),thickness,cv2.LINE_AA)    
    cv2.line(result_image,dot2_start,dot2_end,(255,255,255),thickness,cv2.LINE_AA)  
    
    # 保存输出图片
    return result_image

class ImageToComic:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "remove_background"
    CATEGORY = "image"

    def remove_background(self, image1, image2, image3):
        image_p1 = tensor2pil(image1)
        image_p2 = tensor2pil(image2)
        image_p3 = tensor2pil(image3)
        image = pil2tensor(combine_image(image_p1, image_p2, image_p3))
        return (image,)

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "Image To Comic": ImageToComic
}
