from pycocotools.coco import COCO
import os
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw
import skimage.io as io
import json

"""
路径参数
"""
# 原coco数据集的路径
dataDir = "./data/coco"
# 用于保存新生成的数据的路径
savepath = "data/CropBank/train2017.1@10_instances/"
# 只保存含有你需要的类别的图片的路径，最后没有用
# 因为没必要，coco是按json中的信息读图，只要在json里做筛选就行了
img_save = savepath + "images"
# 最后生产的json文件的保存路径
anno_save = savepath + "Annotations"
""" 
数据集参数
"""
# coco有80类，这里写要提取部分类的名字
# classes_names = ['cat','dog']
classes_names = []
# 要处理的数据集，比如val2017、train2017等
datasets_list = ["./data/coco/train2017"]
# data/coco/annotations/semi_supervised/instances_train2017.1@1.json


# 获取COCO中所有类别名称
def getAllClasses():
    classes_names = []
    with open(
        "./data/coco/annotations/semi_supervised/instances_train2017.1@1.json", "r"
    ) as f:
        coco_data = json.load(f)
    categories = coco_data["categories"]
    for category in categories:
        # print('标签ID:', category['id'], '标签名称:', category['name'])
        classes_names.append(category["name"])
    return classes_names


# 生成保存路径
# if the dir is not exists,make it,else delete it
def mkr(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.mkdir(path)
    else:
        os.mkdir(path)


# 获取并处理所有需要的json数据
def process_json_data(annFile, class_name):
    # 获取COCO_json的数据
    coco = COCO(annFile)
    # 拿到所有需要的图片数据的id
    classes_ids = coco.getCatIds(catNms=class_name)
    # 加载所有需要的类别信息
    classes_list = coco.loadCats(classes_ids)
    # 取所有类别的并集的所有图片id
    # 如果想要交集，不需要循环，直接把所有类别作为参数输入，即可得到所有类别都包含的图片
    imgIds_list = []
    for idx in classes_ids:
        imgidx = coco.getImgIds(catIds=idx)
        imgIds_list += imgidx
    # 去除重复的图片
    imgIds_list = list(set(imgIds_list))
    # 一次性获取所有图像的信息
    image_info_list = coco.loadImgs(imgIds_list)
    # 获取图像中对应类别的分割信息,由catIds来指定
    annIds = coco.getAnnIds(imgIds=[], catIds=classes_ids, iscrowd=0)
    anns_list = coco.loadAnns(annIds)
    return classes_list, image_info_list, anns_list


# 保存数据到json
def save_json_data(json_file, classes_list, image_info_list, anns_list):
    coco_sub = dict()
    coco_sub["info"] = dict()
    coco_sub["licenses"] = []
    coco_sub["images"] = []
    coco_sub["type"] = "instances"
    coco_sub["annotations"] = []
    coco_sub["categories"] = []
    # 以下非必须,为coco数据集的前缀信息
    coco_sub["info"]["description"] = "COCO 2017 sub Dataset"
    coco_sub["info"][
        "url"
    ] = "https://blog.csdn.net/Transfattyacids?spm=1000.2115.3001.5343"
    coco_sub["info"]["version"] = "1.0"
    coco_sub["info"]["year"] = 2023
    coco_sub["info"]["contributor"] = "CStar777"
    coco_sub["info"]["date_created"] = "2023-7-2 10:10"
    sub_license = dict()
    sub_license["url"] = "https://blog.csdn.net/Transfattyacids?spm=1000.2115.3001.5343"
    sub_license["id"] = 1
    sub_license["name"] = "Attribution-NonCommercial-ShareAlike License"
    coco_sub["licenses"].append(sub_license)
    # 以下为必须插入信息,包括image、annotations、categories三个字段
    # 插入image信息
    coco_sub["images"].extend(image_info_list)
    for i in anns_list:
        if i["area"] > 2000:
            # 插入annotation信息
            coco_sub["annotations"].append(i)
    # 插入categories信息
    coco_sub["categories"].extend(classes_list)
    # # 插入image信息
    # coco_sub["images"].extend(image_info_list)
    # # 插入annotation信息
    # coco_sub["annotations"].extend(anns_list)
    # # 插入categories信息
    # coco_sub["categories"].extend(classes_list)
    # 自此所有该插入的数据插入完毕
    # 最后一步，保存数据
    json.dump(coco_sub, open(json_file, "w"))


if __name__ == "__main__":
    mkr(img_save)
    mkr(anno_save)
    classes_names = getAllClasses()
    print(classes_names)
    # 逐类别处理
    for class_name in classes_names:
        # 按单个数据集进行处理
        for dataset in datasets_list:
            # 获取要处理的json文件路径
            annFile = (
                "{}/annotations/semi_supervised/instances_train2017.1@1.json".format(
                    dataDir
                )
            )
            # 存储处理完成的json文件路径
            json_file = "{}/{}.json".format(anno_save, class_name)
            # 处理数据
            classes_list, image_info_list, anns_list = process_json_data(
                annFile, class_name
            )
            # 保存数据
            save_json_data(json_file, classes_list, image_info_list, anns_list)
            print("instances_train2017.1@1_{}.json saved".format(class_name))
