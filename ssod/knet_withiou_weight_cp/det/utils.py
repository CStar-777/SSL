import torch
from PIL import Image
import imgviz
import argparse
import os
import numpy as np
from numpy import random
import tqdm
from skimage import measure
from skimage.io import imread
from pycocotools.coco import COCO
from mmdet.datasets.coco import *
import cv2
import matplotlib.pyplot as plt
from matplotlib import patches
from mmdet.core import BitmapMasks
from ssod.utils.logger import *
from mmcv.image.photometric import imdenormalize, imnormalize
import json
import torch.nn as nn
import torch.nn.functional as F
import re
import time

annot_path = "/data/Datasets_cx/data/coco/annotations/instances_train2017.json"
ann_path = "/data/Datasets_cx/data/CropBank/train2017.1@10_instances/Annotations/"
imgRoot = "/data/Datasets_cx/data/coco/train2017/"
dataType = "train2017"

json_file = "/data/Datasets_cx/data/coco/annotations/instances_train2017.json"
img_file = "/data/Datasets_cx/data/coco/train2017/"
out_file1 = "/data/Datasets_cx/image/tailClass/"
out_file2 = "/data/Datasets_cx/image/sameClass/"
coco = COCO(annot_path)

CLASSES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)


def sem2ins_masks(gt_sem_seg, num_thing_classes=80):
    """Convert semantic segmentation mask to binary masks

    Args:
        gt_sem_seg (torch.Tensor): Semantic masks to be converted.
            [0, num_thing_classes-1] is the classes of things,
            [num_thing_classes:] is the classes of stuff.
        num_thing_classes (int, optional): Number of thing classes.
            Defaults to 80.

    Returns:
        tuple[torch.Tensor]: (mask_labels, bin_masks).
            Mask labels and binary masks of stuff classes.
    """
    # gt_sem_seg is zero-started, where zero indicates the first class
    # since mmdet>=2.17.0, see more discussion in
    # https://mmdetection.readthedocs.io/en/latest/conventions.html#coco-panoptic-dataset  # noqa
    classes = torch.unique(gt_sem_seg)
    # classes ranges from 0 - N-1, where the class IDs in
    # [0, num_thing_classes - 1] are IDs of thing classes
    masks = []
    labels = []

    for i in classes:
        # skip ignore class 255 and "thing classes" in semantic seg
        if i == 255 or i < num_thing_classes:
            continue
        labels.append(i)
        masks.append(gt_sem_seg == i)

    if len(labels) > 0:
        labels = torch.stack(labels)
        masks = torch.cat(masks)
    else:
        labels = gt_sem_seg.new_zeros(size=[0])
        masks = gt_sem_seg.new_zeros(
            size=[0, gt_sem_seg.shape[-2], gt_sem_seg.shape[-1]]
        )
    return labels.long(), masks.float()


def compute_mask_iou(inputs, targets):
    inputs = inputs.sigmoid()
    # thresholding
    binarized_inputs = (inputs >= 0.5).float()
    # print("binarized_inputs", binarized_inputs)
    # targets = (targets > 0.5).float()
    intersection = (binarized_inputs * targets).sum(-1)
    union = targets.sum(-1) + binarized_inputs.sum(-1) - intersection
    score = intersection / (union + 1e-6)
    return score


def gen_masks_from_bboxes(bboxes, img_shape):
    """
    通过bbox生成掩码
        输入：bbox坐标: x, y, w, h; 生成掩码的图像shape
        输出：与输入图像同高宽的，但只有bbox区域是1，其余区域为0的二值mask
    """
    img_h, img_w = img_shape[:2]
    x, y, w, h = bboxes
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    bbox = np.array([[[x, y], [x + w, y], [x + w, y + h], [x, y + h]]], dtype=np.int32)
    mask = cv2.fillPoly(mask, bbox, 1)

    return mask


def is_valid_integer(string):
    pattern = r"^[-+]?\d+$"
    return bool(re.match(pattern, string))


def poly2mask(points, width, height):
    """
    通过多边形坐标生成图像掩码
        输入：segmetation坐标串
        输出：与输入图像同高宽的，但只有实例区域是1，其余区域为0的二值mask
    """
    polys = []
    for seg in points:
        # if is_valid_integer(seg):
        try:
            poly = np.array(seg, dtype=np.int32).reshape((int(len(seg) / 2), 2))
        except ValueError:
            print("多边形坐标生成图像掩码出错")
            mask = np.zeros((width, height), dtype=np.int32)
            return mask
        polys.append(poly)
    mask = np.zeros((width, height), dtype=np.int32)
    cv2.fillPoly(mask, polys, 1)
    return mask


# close_contour/binary_mask_to_polygon/get_paired_coord
# 是实现：从掩膜 mask 转换回多边形 poly  的函数
def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour


def binary_mask_to_polygon(binary_mask, tolerance=0):
    """Converts a binary mask to COCO polygon representation
    Args:
    binary_mask: a 2D binary numpy array where '1's represent the object
    tolerance: Maximum distance from original points of polygon to approximated
    polygonal chain. If tolerance is 0, the original coordinate array is returned.
    """
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(
        binary_mask, pad_width=1, mode="constant", constant_values=0
    )
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)
    return polygons


def get_paired_coord(coord):
    points = None
    for i in range(0, len(coord), 2):
        point = np.array(coord[i : i + 2], dtype=np.int32).reshape(1, 2)
        if points is None:
            points = point
        else:
            points = np.concatenate([points, point], axis=0)
    return points

def copy_paste(i, j, img, teacher_info, det_masks, classes_thr, type):
    """
    输入:原图像文件名,伪标签预测类别名称,预测伪mask

    """
    # start = time.time()
    # dst_mask = teacher_info['det_masks'][i][j]
    det_mask = det_masks[i].masks
    label = teacher_info["det_labels"][i][j]
    dst_img_tensor = img[i].cpu().detach()
    thrs = classes_thr.tolist()
    if len(thrs) < 80:
        type = 2
    # 看看用img_id读出来是什么样子
    dst_img_fname = teacher_info["img_metas"][i]["filename"]
    dst_file_path = dst_img_fname.split("/")[-1]
    dst_file_name, file_ext = os.path.splitext(dst_file_path)  # 获取图像编号，用来后面命名合成图像
    # dst_img0 = imread("/home/hujie/hujie-project/SoftTeacher/" + dst_img_fname)
    # 对原图像进行还原（逆归一化操作）
    mean = teacher_info["img_metas"][i]["img_norm_cfg"]["mean"]
    std = teacher_info["img_metas"][i]["img_norm_cfg"]["std"]
    dst_img = color_transform(img_tensor=dst_img_tensor, mean=mean, std=std)
    _, _, dst_h, dst_w = img.shape  # (896,1344)
    det_h, det_w = det_masks[i].height, det_masks[i].width
    # t1 = time.time() - start

    if type == 1:
        class_min = thrs.index(min(thrs))
        class_name = CLASSES[int(class_min)]
    else:
        # 通过label(int)得类名
        class_name = CLASSES[int(label)]
    # 由于label与数据集中的类别id不对应，再通过类名得数据集中对应的类别id
    # class_id = coco.get_cat_ids(cat_names=class_name)
    # t2 = time.time() - start - t1

    # 预测类的JSON文件路径
    ann_filename = ann_path + class_name + ".json"
    # 加载JSON文件
    with open(ann_filename, "r") as f:
        instance_data = json.load(f)
    # 随机抽取JSON文件中同类别crop
    ann_num = len(instance_data["annotations"])
    temp = random.randint(0, ann_num)  # 随机选择
    src_segmentation = instance_data["annotations"][temp]["segmentation"]
    src_img_id = instance_data["annotations"][temp]["image_id"]
    # t3 = time.time() - start - t2 - t1

    src_img_ = coco.loadImgs(src_img_id)[0]
    # src_img = imread(img_file + src_img_['file_name']) # RGB读入
    src_img = cv2.imread(img_file + src_img_["file_name"])  # BGR读入
    src_h, src_w, _ = src_img.shape
    # 将src的多边形坐标转换为二值mask
    src_mask = poly2mask(src_segmentation, src_h, src_w)
    if np.all(src_mask == 0):
        return img, teacher_info, det_masks

    # 调整scr_img和scr_mask的尺寸，与teacher['img_shape']的大小一致
    src_img_rs = cv2.resize(src_img, (det_w, det_h), interpolation=cv2.INTER_NEAREST)
    src_mask_rs = cv2.resize(src_mask, (det_w, det_h), interpolation=cv2.INTER_NEAREST)
    # 将src_img, padding到img.shape的大小（batch_input_shape）
    # pad_H, pad_W = teacher_info["img_metas"][i]["batch_input_shape"]
    pad_wh = (0, dst_w - det_w, 0, dst_h - det_h)
    pad_cwh = (0, 0, 0, dst_w - det_w, 0, dst_h - det_h)
    src_img_rs = torch.from_numpy(src_img_rs).float().cuda()
    src_mask_rs = torch.from_numpy(src_mask_rs).float().cuda()
    src_img_rs_pad = F.pad(src_img_rs, pad_cwh, value=0)
    src_mask_rs_pad = F.pad(src_mask_rs, pad_wh, value=0)
    src_img_rs_pad = src_img_rs_pad.cpu().numpy()
    src_mask_rs_pad = src_mask_rs_pad.cpu().numpy()

    # 对dst_img中被遮挡的实例更新掩码和标签
    det_mask = np.where(src_mask_rs.cpu().numpy(), 0, det_mask)
    # t4 = time.time() - start - t3 - t2 - t1

    # 将选择粘贴的实例的像素点叠加到图片上
    img_new = (
        dst_img * (1 - src_mask_rs_pad[..., np.newaxis])
        + src_img_rs_pad * src_mask_rs_pad[..., np.newaxis]
    )
    # 存下来看看
    if type == 1:
        mmcv.imwrite(img_new, out_file1 + dst_file_name + "_test.jpg")
    # else:
    #     mmcv.imwrite(img_new, out_file2 + dst_file_name + "_test.jpg")
    # t5 = time.time() - start - t4 - t3 - t2 - t1

    # 更新与整合
    # 对更新后的img进行归一化操作（这里转来转去的，不确定格式有没有出错）
    img_new = imnormalize(img=img_new, mean=mean, std=std)
    # mask_tensor = gt_mask.to_tensor(torch.float, gt_labels[0].device)
    img_new = img_new.transpose((2, 0, 1))
    img_new_tensor = torch.from_numpy(img_new).to(label.device)
    img[i] = img_new_tensor

    # 更新teacher_info中的det_labels, det_iou, det_scores, det_masks
    det_label_new = np.concatenate(
        [teacher_info["det_labels"][i].cpu().numpy(), [label.cpu().numpy()]]
    )
    det_labels_list = list(teacher_info["det_labels"])
    det_labels_list[i] = torch.from_numpy(det_label_new).to(label.device)
    teacher_info["det_labels"] = tuple(det_labels_list)

    det_iou_new = np.concatenate([teacher_info["det_ious"][i].cpu().numpy(), [1.0000]])
    det_iou_list = list(teacher_info["det_ious"])
    det_iou_list[i] = torch.from_numpy(det_iou_new).float().to(label.device)
    teacher_info["det_ious"] = tuple(det_iou_list)

    det_scores_new = np.concatenate(
        [teacher_info["det_scores"][i].cpu().numpy(), [1.0000]]
    )
    det_scores_list = list(teacher_info["det_scores"])
    det_scores_list[i] = torch.from_numpy(det_scores_new).float().to(label.device)
    teacher_info["det_scores"] = tuple(det_scores_list)

    det_mask = np.concatenate([det_mask, [src_mask_rs.cpu().numpy()]])
    det_masks[i].masks = det_mask
    # t6 = time.time() - start - t5 - t4 - t3 - t2 - t1

    # end = time.time() - start
    # print(
    #     f"各阶段用时分别为:{t1:.2f}s,{t2:.2f}s,{t3:.2f}s,{t4:.2f}s,{t5:.2f}s,{t6:.2f}s,总共用时:{end:.2f}s"
    # )

    return img, teacher_info, det_masks
