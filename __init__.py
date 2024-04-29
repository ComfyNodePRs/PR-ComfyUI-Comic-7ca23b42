from PIL import ImageFont, Image, ImageDraw
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

def draw_text(text,image,font_path):
    min_font_size = 20
    max_font_size = 80
    region_width = 280
    region_height = 100

    text_image4 = image.copy()
    pil_image = Image.fromarray(text_image4)
    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.truetype(font_path, min_font_size)
    text_width, text_height = draw.textsize(text, font)
   
    if text_width <= region_width * 2:
        if text_width > region_width:
            # 两行显示
            max_word = region_width // min_font_size - 1
            first_text = ''.join(text[:max_word])
            for index, word in enumerate(text[max_word:]):
                if draw.textlength(first_text + word, font) <= region_width:
                    first_text += word
                else:
                    second_text = text[index + max_word:]
                    break

            draw.text((100, 70+25+10 + min_font_size/2-int(draw.textsize(text, font)[1])), first_text, font=font, fill=(0, 0, 0))
            draw.text((100, 70+25+50-10 + min_font_size/2-int(draw.textsize(text, font)[1])), second_text, font=font, fill=(0, 0, 0))
        else:
            # 一行显示
            font_size = max_font_size
            while True:
                font = ImageFont.truetype(font_path, font_size)
                if draw.textlength(text, font) <= region_width:
                    break
                else:
                    font_size -= 2
                
            draw.text((100,font_size/2-int(draw.textsize(text, font)[1])+70+50), text, font=font, fill=(0, 0, 0))
        text_image4 = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGBA2BGRA)
        return text_image4
    else:
        print("字数超出限制！")

def rotate_image(image,angle):
    height, width = image.shape[:2]
    center_x = width // 2
    center_y = height // 2
    M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
    rotate_image = cv2.warpAffine(image, M, (width, height))
    return rotate_image
    
# 将多个分镜合并成一张漫画
def combine_image(image_p1, image_p2, image_p3, text):

    # 分割斜线
    dot1_start = (0,512)
    dot1_end = (1024,400)
    dot2_start = ((int((dot1_start[0] + dot1_end[0]) / 2)), int((dot1_start[1] + dot1_end[1]) / 2))
    dot2_end = (600,1024)

    # 分割线宽度
    thickness = 10

    # 读取对话框图片
    dialog_name = "dialog.png"
    dialog_folder = "assets"
    dialog_file = os.path.join(dialog_folder, dialog_name)
    resolved_dialog_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), dialog_file)
    image4_origin = cv2.imread(resolved_dialog_path, cv2.IMREAD_UNCHANGED)

    image4_origin = cv2.resize(image4_origin,(513, 273))
    image4_origin = rotate_image(image4_origin,350)

    font_name = "TencentSans-W3.ttf"
    font_folder = "fonts"
    font_file = os.path.join(font_folder, font_name)
    resolved_font_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), font_file)
    image4_origin = draw_text(text,image4_origin, str(resolved_font_path))
    image4_origin = rotate_image(image4_origin,-350)

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

    image4 = np.zeros((1024, 1024, 4), dtype=np.uint8)
    x_offset = 400
    y_offset = 300
    image4[y_offset:y_offset + 273 , x_offset:x_offset + 513] = image4_origin

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

    # 绘制对话框
    alpha_mask = image4[:, :, 3]
    for c in range(3):
        result_image[:, :, c] = (image4[:, :, c] * (alpha_mask / 255.0) + result_image[:, :, c] * (1 - alpha_mask / 255.0))
    
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
    FUNCTION = "image2comic"
    CATEGORY = "image"

    def image2comic(self, image1, image2, image3):
        image_p1 = tensor2pil(image1)
        image_p2 = tensor2pil(image2)
        image_p3 = tensor2pil(image3)
        image = pil2tensor(combine_image(image_p1, image_p2, image_p3, "五星上将麦克阿瑟曾经说过"))
        return (image,)

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "Image To Comic": ImageToComic
}
