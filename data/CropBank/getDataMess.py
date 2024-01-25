from pycocotools.coco import COCO
import os
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw
import skimage.io as io
import json


global cnt


# 加载JSON文件
# data/coco/annotations/semi_supervised/instances_train2017.1@1.json
# ./data/CropBank/train2017.1@10_instances/Annotations/instances_train2017.1@1_cat.json
with open('./data/coco/annotations/instances_train2017.json', 'r') as f:
    coco_data = json.load(f)
    
cnt = 1
# 获取标签信息
categories = coco_data['categories']
print('标签数量：', len(categories))
# 打印每个标签的名称和ID
for category in categories:
    print('标签ID:', category['id'], '标签名称:', category['name'])

'''
# 获取注释信息
annotations = coco_data['annotations']
print('注释数量：', len(annotations))
input("随意输入，打印每个注释的目标ID和边界框")    
# 打印每个注释的目标ID和边界框
for annotation in annotations:
    print(str(cnt)+'.目标ID:', annotation['id'], '边界框:', annotation['bbox'])
    cnt = cnt+1
'''    
