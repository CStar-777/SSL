# Copyright (c) OpenMMLab. All rights reserved.
import copy
import inspect
import math
import warnings

import cv2
import mmcv
import numpy as np
from numpy import random

from mmdet.core import BitmapMasks, PolygonMasks, find_inside_bboxes
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from mmdet.utils import log_img_scale
# from ..builder import PIPELINES
from mmdet.datasets import PIPELINES

try:
    from imagecorruptions import corrupt
except ImportError:
    corrupt = None

try:
    import albumentations
    from albumentations import Compose
except ImportError:
    albumentations = None
    Compose = None


# cx add
@PIPELINES.register_module()
class CopyPaste_v0:
    """Simple Copy-Paste is a Strong Data Augmentation Method for Instance
    Segmentation The simple copy-paste transform steps are as follows:
    1. The destination image is already resized with aspect ratio kept,
       cropped and padded.
    2. Randomly select a source image, which is also already resized
       with aspect ratio kept, cropped and padded in a similar way
       as the destination image.
    3. Randomly select some objects from the source image.
    4. Paste these source objects to the destination image directly,
       due to the source and destination image have the same size.
    5. Update object masks of the destination image, for some origin objects
       may be occluded.
    6. Generate bboxes from the updated destination masks and
       filter some objects which are totally occluded, and adjust bboxes
       which are partly occluded.
    7. Append selected source bboxes, masks, and labels.
    Args:
        max_num_pasted (int): The maximum number of pasted objects.
            Default: 100.
        bbox_occluded_thr (int): The threshold of occluded bbox.
            Default: 10.
        mask_occluded_thr (int): The threshold of occluded mask.
            Default: 300.
        selected (bool): Whether select objects or not. If select is False,
            all objects of the source image will be pasted to the
            destination image.
            Default: True.
    """

    def __init__(
        self,
        max_num_pasted=100,
        bbox_occluded_thr=10,
        mask_occluded_thr=300,
        selected=True,
    ):
        self.max_num_pasted = max_num_pasted
        self.bbox_occluded_thr = bbox_occluded_thr
        self.mask_occluded_thr = mask_occluded_thr
        self.selected = selected
        self.paste_by_box = False

    # get_indexes:表明当前数据增强需要随机选取其他图片进行辅助
    def get_indexes(self, dataset):
        """Call function to collect indexes.s.
        Args:
            dataset (:obj:`MultiImageMixDataset`): The dataset.
        Returns:
            list: Indexes.
        """
        return random.randint(0, len(dataset))

    def gen_masks_from_bboxes(self, bboxes, img_shape):
        """Generate gt_masks based on gt_bboxes.
        Args:
            bboxes (list): The bboxes's list.
            img_shape (tuple): The shape of image.
        Returns:
            BitmapMasks
        """
        self.paste_by_box = True
        img_h, img_w = img_shape[:2]
        xmin, ymin = bboxes[:, 0:1], bboxes[:, 1:2]
        xmax, ymax = bboxes[:, 2:3], bboxes[:, 3:4]
        gt_masks = np.zeros((len(bboxes), img_h, img_w), dtype=np.uint8)
        for i in range(len(bboxes)):
            gt_masks[i,
                     int(ymin[i]):int(ymax[i]),
                     int(xmin[i]):int(xmax[i])] = 1
        return BitmapMasks(gt_masks, img_h, img_w)

    def get_gt_masks(self, results):
        """Get gt_masks originally or generated based on bboxes.
        If gt_masks is not contained in results,
        it will be generated based on gt_bboxes.
        Args:
            results (dict): Result dict.
        Returns:
            BitmapMasks: gt_masks, originally or generated based on bboxes.
        """
        if results.get('gt_masks', None) is not None:
            return results['gt_masks']
        else:
            return self.gen_masks_from_bboxes(
                results.get('gt_bboxes', []), results['img'].shape)

    def __call__(self, results):
        """Call function to make a copy-paste of image.
        Args:
            results (dict): Result dict.
        Returns:
            dict: Result dict with copy-paste transformed.
        """

        assert 'mix_results' in results
        num_images = len(results['mix_results'])
        assert num_images == 1, \
            f'CopyPaste only supports processing 2 images, got {num_images}'

        # Get gt_masks originally or generated based on bboxes.
        results['gt_masks'] = self.get_gt_masks(results)
        # only one mix picture
        results['mix_results'][0]['gt_masks'] = self.get_gt_masks(
            results['mix_results'][0])

        if self.selected:
            selected_results = self._select_object(results['mix_results'][0])
        else:
            selected_results = results['mix_results'][0]
        return self._copy_paste(results, selected_results)

    def _select_object(self, results):
        """Select some objects from the source results."""
        bboxes = results['gt_bboxes']
        labels = results['gt_labels']
        masks = results['gt_masks']
        max_num_pasted = min(bboxes.shape[0] + 1, self.max_num_pasted)
        num_pasted = np.random.randint(0, max_num_pasted)
        selected_inds = np.random.choice(
            bboxes.shape[0], size=num_pasted, replace=False)

        selected_bboxes = bboxes[selected_inds]
        selected_labels = labels[selected_inds]
        selected_masks = masks[selected_inds]

        results['gt_bboxes'] = selected_bboxes
        results['gt_labels'] = selected_labels
        results['gt_masks'] = selected_masks
        return results

    def _copy_paste(self, dst_results, src_results):
        """CopyPaste transform function.
        Args:
            dst_results (dict): Result dict of the destination image.
            src_results (dict): Result dict of the source image.
        Returns:
            dict: Updated result dict.
        """
        dst_img = dst_results['img']
        dst_bboxes = dst_results['gt_bboxes']
        dst_labels = dst_results['gt_labels']
        dst_masks = dst_results['gt_masks']

        src_img = src_results['img']
        src_bboxes = src_results['gt_bboxes']
        src_labels = src_results['gt_labels']
        src_masks = src_results['gt_masks']

        if len(src_bboxes) == 0:
            if self.paste_by_box:
                dst_results.pop('gt_masks')
            return dst_results

        # update masks and generate bboxes from updated masks
        # 对原图中被遮挡的实例更新掩码和标签
        composed_mask = np.where(np.any(src_masks.masks, axis=0), 1, 0)
        updated_dst_masks = self.get_updated_masks(dst_masks, composed_mask)
        updated_dst_bboxes = updated_dst_masks.get_bboxes()
        assert len(updated_dst_bboxes) == len(updated_dst_masks)

        # filter totally occluded objects
        # 通过 bbox_occluded_thr 和 mask_occluded_thr 过滤掉不符合条件的实例
        bboxes_inds = np.all(
            np.abs(
                (updated_dst_bboxes - dst_bboxes)) <= self.bbox_occluded_thr,
            axis=-1)
        masks_inds = updated_dst_masks.masks.sum(
            axis=(1, 2)) > self.mask_occluded_thr
        valid_inds = bboxes_inds | masks_inds

        # Paste source objects to destination image directly
        # 将选择粘贴的实例的像素点叠加到图片上，并将粘贴的实例的标签与更新后的实例的标签整合到一起
        img = dst_img * (1 - composed_mask[..., np.newaxis]
                         ) + src_img * composed_mask[..., np.newaxis]
        bboxes = np.concatenate([updated_dst_bboxes[valid_inds], src_bboxes])
        labels = np.concatenate([dst_labels[valid_inds], src_labels])
        masks = np.concatenate(
            [updated_dst_masks.masks[valid_inds], src_masks.masks])

        dst_results['img'] = img
        dst_results['gt_bboxes'] = bboxes
        dst_results['gt_labels'] = labels
        if self.paste_by_box:
            dst_results.pop('gt_masks')
        else:
            dst_results['gt_masks'] = BitmapMasks(masks, masks.shape[1],
                                                  masks.shape[2])

        return dst_results

    # 原图实例掩码更新
    def get_updated_masks(self, masks, composed_mask):
        assert masks.masks.shape[-2:] == composed_mask.shape[-2:], \
            'Cannot compare two arrays of different size'
        masks.masks = np.where(composed_mask, 0, masks.masks)
        return masks

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'max_num_pasted={self.max_num_pasted}, '
        repr_str += f'bbox_occluded_thr={self.bbox_occluded_thr}, '
        repr_str += f'mask_occluded_thr={self.mask_occluded_thr}, '
        repr_str += f'selected={self.selected}, '
        return repr_str

# 修改CopyPaste
@PIPELINES.register_module()
class CopyPaste_v2:
    def __init__(
        self,
        max_num_pasted=100,
        bbox_occluded_thr=10,
        mask_occluded_thr=300,
        selected=True,
    ):
        self.max_num_pasted = max_num_pasted
        self.bbox_occluded_thr = bbox_occluded_thr
        self.mask_occluded_thr = mask_occluded_thr
        self.selected = selected
        self.paste_by_box = False

    # get_indexes:表明当前数据增强需要随机选取其他图片进行辅助
    # 修改：在指定类的数据集中随机
    def get_indexes(self, dataset):
        """Call function to collect indexes.s.
        Args:
            dataset (:obj:`MultiImageMixDataset`): The dataset.
        Returns:
            list: Indexes.
        """
        return random.randint(0, len(dataset))

    def gen_masks_from_bboxes(self, bboxes, img_shape):
        """Generate gt_masks based on gt_bboxes.
        Args:
            bboxes (list): The bboxes's list.
            img_shape (tuple): The shape of image.
        Returns:
            BitmapMasks
        """
        self.paste_by_box = True
        img_h, img_w = img_shape[:2]
        xmin, ymin = bboxes[:, 0:1], bboxes[:, 1:2]
        xmax, ymax = bboxes[:, 2:3], bboxes[:, 3:4]
        gt_masks = np.zeros((len(bboxes), img_h, img_w), dtype=np.uint8)
        for i in range(len(bboxes)):
            gt_masks[i,
                     int(ymin[i]):int(ymax[i]),
                     int(xmin[i]):int(xmax[i])] = 1
        return BitmapMasks(gt_masks, img_h, img_w)

    def get_gt_masks(self, results):
        """Get gt_masks originally or generated based on bboxes.
        If gt_masks is not contained in results,
        it will be generated based on gt_bboxes.
        Args:
            results (dict): Result dict.
        Returns:
            BitmapMasks: gt_masks, originally or generated based on bboxes.
        """
        if results.get('gt_masks', None) is not None:
            return results['gt_masks']
        else:
            return self.gen_masks_from_bboxes(
                results.get('gt_bboxes', []), results['img'].shape)

    def __call__(self, results):
        """Call function to make a copy-paste of image.
        Args:
            results (dict): Result dict.
        Returns:
            dict: Result dict with copy-paste transformed.
        """

        assert 'mix_results' in results
        num_images = len(results['mix_results'])
        assert num_images == 1, \
            f'CopyPaste only supports processing 2 images, got {num_images}'

        # Get gt_masks originally or generated based on bboxes.
        results['gt_masks'] = self.get_gt_masks(results)
        # only one mix picture
        results['mix_results'][0]['gt_masks'] = self.get_gt_masks(
            results['mix_results'][0])

        if self.selected:
            selected_results = self._select_object(results['mix_results'][0])
        else:
            selected_results = results['mix_results'][0]
        return self._copy_paste(results, selected_results)

    # cx edit 在选择copyPaste的区域时不能随机
    # dst_bboxes = dst_results['gt_bboxes']
    # 选择生成伪mask(转为box)作为bboxes
    def _select_object(self, results):
        """Select some objects from the source results."""
        bboxes = results['gt_bboxes']
        labels = results['gt_labels']
        masks = results['gt_masks']
        max_num_pasted = min(bboxes.shape[0] + 1, self.max_num_pasted)
        # num_pasted = np.random.randint(0, max_num_pasted)
        print("图像预测box数量：",bboxes.shape[0])
        num_pasted = self.copyPaste_num
        selected_inds = np.random.choice(bboxes.shape[0], size=num_pasted, replace=False)

        selected_bboxes = bboxes[selected_inds]
        selected_labels = labels[selected_inds]
        selected_masks = masks[selected_inds]

        results['gt_bboxes'] = selected_bboxes
        results['gt_labels'] = selected_labels
        results['gt_masks'] = selected_masks
        return results

    def _copy_paste(self, dst_results, src_results):
        """CopyPaste transform function.
        Args:
            dst_results (dict): Result dict of the destination image.
            src_results (dict): Result dict of the source image.
        Returns:
            dict: Updated result dict.
        """
        dst_img = dst_results['img']
        dst_bboxes = dst_results['gt_bboxes']
        dst_labels = dst_results['gt_labels']
        dst_masks = dst_results['gt_masks']

        src_img = src_results['img']
        src_bboxes = src_results['gt_bboxes']
        src_labels = src_results['gt_labels']
        src_masks = src_results['gt_masks']

        if len(src_bboxes) == 0:
            if self.paste_by_box:
                dst_results.pop('gt_masks')
            return dst_results

        # update masks and generate bboxes from updated masks
        # 对原图中被遮挡的实例更新掩码和标签
        composed_mask = np.where(np.any(src_masks.masks, axis=0), 1, 0)
        updated_dst_masks = self.get_updated_masks(dst_masks, composed_mask)
        updated_dst_bboxes = updated_dst_masks.get_bboxes()
        assert len(updated_dst_bboxes) == len(updated_dst_masks)

        # filter totally occluded objects
        # 通过 bbox_occluded_thr 和 mask_occluded_thr 过滤掉不符合条件的实例
        bboxes_inds = np.all(
            np.abs((updated_dst_bboxes - dst_bboxes)) <= self.bbox_occluded_thr,
            axis=-1)
        masks_inds = updated_dst_masks.masks.sum(axis=(1, 2)) > self.mask_occluded_thr
        valid_inds = bboxes_inds | masks_inds

        # Paste source objects to destination image directly
        # 将选择粘贴的实例的像素点叠加到图片上，并将粘贴的实例的标签与更新后的实例的标签整合到一起
        img = dst_img * (1 - composed_mask[..., np.newaxis]) + src_img * composed_mask[..., np.newaxis]
        bboxes = np.concatenate([updated_dst_bboxes[valid_inds], src_bboxes])
        labels = np.concatenate([dst_labels[valid_inds], src_labels])
        masks = np.concatenate([updated_dst_masks.masks[valid_inds], src_masks.masks])

        dst_results['img'] = img
        dst_results['gt_bboxes'] = bboxes
        dst_results['gt_labels'] = labels
        if self.paste_by_box:
            dst_results.pop('gt_masks')
        else:
            dst_results['gt_masks'] = BitmapMasks(masks, masks.shape[1],
                                                  masks.shape[2])

        return dst_results

    # 原图实例掩码更新
    def get_updated_masks(self, masks, composed_mask):
        assert masks.masks.shape[-2:] == composed_mask.shape[-2:], \
            'Cannot compare two arrays of different size'
        masks.masks = np.where(composed_mask, 0, masks.masks)
        return masks

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'max_num_pasted={self.max_num_pasted}, '
        repr_str += f'bbox_occluded_thr={self.bbox_occluded_thr}, '
        repr_str += f'mask_occluded_thr={self.mask_occluded_thr}, '
        repr_str += f'selected={self.selected}, '
        return repr_str

