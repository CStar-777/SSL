import torch
import numpy as np

def sem2ins_masks(gt_sem_seg,
                  num_thing_classes=80):
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
            size=[0, gt_sem_seg.shape[-2], gt_sem_seg.shape[-1]])
    return labels.long(), masks.float()


def compute_mask_iou(inputs, targets):
    inputs = inputs.sigmoid()
    # thresholding 
    binarized_inputs = (inputs >= 0.5).float()
    #print("binarized_inputs", binarized_inputs)
    #targets = (targets > 0.5).float()
    intersection = (binarized_inputs * targets).sum(-1)
    union = targets.sum(-1) + binarized_inputs.sum(-1) - intersection
    score = intersection / (union + 1e-6)
    return score


def compute_multi_mask_iou(inputs, targets):
    inputs = inputs.sigmoid()
    # thresholding 
    thr = [0.05+0.05*i for i in range(2, 18)]
    binarized_inputs = (inputs > 0.1).float()
    #print("binarized_inputs", binarized_inputs)
    #targets = (targets > 0.5).float()
    intersection = (binarized_inputs * targets).sum(-1)
    union = targets.sum(-1) + binarized_inputs.sum(-1) - intersection
    multi_score = intersection / (union + 1e-6)
    multi_score = multi_score.unsqueeze(1)
    for t in thr:
        binarized_inputs = (inputs > t).float()
        #print("binarized_inputs", binarized_inputs)
        #targets = (targets > 0.5).float()
        intersection = (binarized_inputs * targets).sum(-1)
        union = targets.sum(-1) + binarized_inputs.sum(-1) - intersection
        score = intersection / (union + 1e-6)
        score = score.unsqueeze(1)
        #print("multi_score", multi_score.shape)
        #print("score", score.shape)
        multi_score = torch.cat((multi_score, score), dim=1)
    return multi_score


def find_max_ind(lst):
    num_mask, num_thr = lst.shape
    lst_thr = []
    for i in range(num_mask):
        max_ind = 0
        maxa = lst[i][0]
        for j in range(1, num_thr):
            if lst[i][j] > maxa:
                maxa = lst[i][j]
                max_ind = j
        lst_thr.append(max_ind)
    thr_ind = np.array(lst_thr)
    return thr_ind