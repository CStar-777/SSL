from pycocotools.coco import COCO
import os
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw
import skimage.io as io
import json

def id2className(num):
    '''
    输入:COCO类别id(0~79)
    输出:对应类别名称
    '''
    class_id = num
    
    if class_id == 0:   # 从0开始，但COCO的id编号从1开始
        class_name = 'person'
    elif class_id == 1:
        class_name = 'bicycle'
    elif class_id == 2:
        class_name = 'car'
    elif class_id == 3:
        class_name = 'motorcycle'
    elif class_id == 4:
        class_name = 'airplane'
    elif class_id == 5:
        class_name = 'bus'
    elif class_id == 6:
        class_name = 'train'
    elif class_id == 7:
        class_name = 'truck'
    elif class_id == 8:
        class_name = 'boat'
    elif class_id == 9:
        class_name = 'traffic light'
    elif class_id == 10:
        class_name = 'fire hydrant'
    elif class_id == 12:    # 没有id=12的标签
        class_name = 'stop sign'
    elif class_id == 13:
        class_name = 'parking meter'
    elif class_id == 14:
        class_name = 'bench'
    elif class_id == 15:
        class_name = 'bird'
    elif class_id == 16:
        class_name = 'cat'
    elif class_id == 17:
        class_name = 'dog'
    elif class_id == 18:
        class_name = 'horse'
    elif class_id == 19:
        class_name = 'sheep'
    elif class_id == 20:
        class_name = 'cow'
    elif class_id == 21:
        class_name = 'elephant'
    elif class_id == 22:
        class_name = 'bear'
    elif class_id == 23:
        class_name = 'zebra'
    elif class_id == 24:
        class_name = 'giraffe'
    elif class_id == 26:
        class_name = 'backpack'
    elif class_id == 27:
        class_name = 'umbrella'
    elif class_id == 30:
        class_name = 'handbag'
    elif class_id == 31:
        class_name = 'tie'
    elif class_id == 32:
        class_name = 'suitcase'
    elif class_id == 33:
        class_name = 'frisbee'
    elif class_id == 34:
        class_name = 'skis'
    elif class_id == 35:
        class_name = 'snowboard'
    elif class_id == 36:
        class_name = 'sports ball'
    elif class_id == 37:
        class_name = 'kite'
    elif class_id == 38:
        class_name = 'baseball bat'
    elif class_id == 39:
        class_name = 'baseball glove'
    elif class_id == 40:
        class_name = 'skateboard'
    elif class_id == 41:
        class_name = 'surfboard'
    elif class_id == 42:
        class_name = 'tennis racket'
    elif class_id == 43:
        class_name = 'bottle'
    elif class_id == 45:
        class_name = 'wine glass'
    elif class_id == 46:
        class_name = 'cup'
    elif class_id == 47:
        class_name = 'fork'
    elif class_id == 48:
        class_name = 'knife'
    elif class_id == 49:
        class_name = 'spoon'
    elif class_id == 50:
        class_name = 'bowl'
    elif class_id == 51:
        class_name = 'banana'
    elif class_id == 52:
        class_name = 'apple'
    elif class_id == 53:
        class_name = 'sandwich'
    elif class_id == 54:
        class_name = 'orange'
    elif class_id == 55:
        class_name = 'broccoli'
    elif class_id == 56:
        class_name = 'carrot'
    elif class_id == 57:
        class_name = 'hot dog'
    elif class_id == 58:
        class_name = 'pizza'
    elif class_id == 59:
        class_name = 'donut'
    elif class_id == 60:
        class_name = 'cake'
    elif class_id == 61:
        class_name = 'chair'
    elif class_id == 62:
        class_name = 'couch'
    elif class_id == 63:
        class_name = 'potted plant'
    elif class_id == 64:
        class_name = 'bed'
    elif class_id == 66:
        class_name = 'dining table'
    elif class_id == 69:
        class_name = 'toilet'
    elif class_id == 71:
        class_name = 'tv'
    elif class_id == 72:
        class_name = 'laptop'
    elif class_id == 73:
        class_name = 'mouse'
    elif class_id == 74:
        class_name = 'remote'
    elif class_id == 75:
        class_name = 'keyboard'
    elif class_id == 76:
        class_name = 'cell phone'
    elif class_id == 77:
        class_name = 'microwave'
    elif class_id == 78:
        class_name = 'oven'
    elif class_id == 79:
        class_name = 'toaster'
    elif class_id == 80:
        class_name = 'sink'
    elif class_id == 81:
        class_name = 'refrigerator'
    elif class_id == 83:
        class_name = 'book'
    elif class_id == 84:
        class_name = 'clock'
    elif class_id == 85:
        class_name = 'vase'
    elif class_id == 86:
        class_name = 'scissors'
    elif class_id == 87:
        class_name = 'teddy bear'
    elif class_id == 88:
        class_name = 'hair drier'
    elif class_id == 89:
        class_name = 'toothbrush'

    return class_name

def name2annFilename(name):
    '''
    输入：伪标签类别名称
    输出：对应类别的标注信息文件路径
    '''
    class_name = name
    ann_dir = './data/CropBank/train2017.1@10_instances/Annotations/instances_train2017.1@1_'+class_name+'.json'
    return ann_dir