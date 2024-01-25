#原版设定
import torch
import numpy as np
from mmcv.runner.fp16_utils import force_fp32
from mmdet.core import bbox2roi, multi_apply, mask_matrix_nms_three_steps_dythr
from mmdet.models import DETECTORS, build_detector

from ssod.utils.structure_utils import dict_split, weighted_loss
from ssod.utils import log_image_with_boxes, log_every_n, log_image_with_masks, log_image_with_masks_without_box, isVisualbyCount_return_count, isVisualbyCount

try:
    import sklearn.mixture as skm
except ImportError:
    skm = None

from .multi_stream_detector import MultiSteamDetector
from .utils import Transform2D, filter_invalid_3, filter_invalid_stage1
from mmdet.core.mask.structures import BitmapMasks
import torch.nn.functional as F
import datetime
import time
import matplotlib as mpl
mpl.use('Agg')


@DETECTORS.register_module()
class SslKnet_weight_three_steps_cs(MultiSteamDetector):
    def __init__(self, model: dict, train_cfg=None, test_cfg=None):
        super(SslKnet_weight_three_steps_cs, self).__init__(
            dict(teacher=build_detector(model), student=build_detector(model)),
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
        if train_cfg is not None:
            self.freeze("teacher")
            self.unsup_weight = self.train_cfg.unsup_weight

        num_classes = self.teacher.rpn_head.num_classes
        num_scores = 100
        self.register_buffer(
            'scores', torch.zeros((num_classes, num_scores)))
        self.register_buffer(
            'score_thresholds', torch.empty((8)).fill_(0.35))
        self.iter = 0
    
    def set_iter(self, step):
        self.iter = step

    def forward_train(self, img, img_metas, **kwargs):
        super().forward_train(img, img_metas, **kwargs)
        # 假设按照 1:4 混合，img 的 shape 是 [5,3,h,w]
        kwargs.update({"img": img})
        kwargs.update({"img_metas": img_metas})
        kwargs.update({"tag": [meta["tag"] for meta in img_metas]})
        # 分成3组：有标签，无标签学生，无标签教师，每组都包括 img img_metas gt_bbox等
        data_groups = dict_split(kwargs, "tag")
        for _, v in data_groups.items():
            v.pop("tag")

        loss = {}
        #! Warnings: By splitting losses for supervised data and unsupervised data with different names,
        #! it means that at least one sample for each group should be provided on each gpu.
        #! In some situation, we can only put one image per gpu, we have to return the sum of loss
        #! and log the loss with logger instead. Or it will try to sync tensors don't exist.
        if "sup" in data_groups:
            # 有标签分支正常训练
            '''
            gt_bboxes = data_groups["sup"]["gt_bboxes"]
            log_every_n(
                {"sup_gt_num": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)}
            )
            '''
#---------------------------------------# 
            #print("data_groups['sup']", data_groups["sup"]) #img_metas中有sup属性
    
            #print(data_groups["sup"]["img"])
            #data_groups["sup"]["img"] = data_groups["sup"]["img"].float()
            #print(data_groups["sup"]["img"])
            #print(data_groups["sup"]["img"].shape)
            #start1 = time.time()
            sup_loss = self.student.forward_train(**data_groups["sup"])
            """
            iou_loss = sup_loss["s0_loss_iou"]
            sup_loss["s0_loss_iou"] = iou_loss * 0
            iou_loss = sup_loss["s1_loss_iou"]
            sup_loss["s1_loss_iou"] = iou_loss * 0        
            iou_loss = sup_loss["s2_loss_iou"]
            sup_loss["s2_loss_iou"] = iou_loss * 0 
            """
            #end1 = time.time()
            #print("sup_cost:", end1 - start1)
            sup_loss = {"sup_" + k: v for k, v in sup_loss.items()}
            loss.update(**sup_loss)

        if self.iter < self.train_cfg.get('warmup_step', -1):
            unsup_weight = 0
        else:
            unsup_weight = self.unsup_weight
        if "unsup_student" in data_groups:
            # 无标签分支训练
            unsup_loss = weighted_loss(
                self.foward_unsup_train(
                    data_groups["unsup_teacher"], data_groups["unsup_student"]
                ),
                weight=unsup_weight,
            )
            unsup_loss = {"unsup_" + k: v for k, v in unsup_loss.items()}
            loss.update(**unsup_loss)

        return loss

    def foward_unsup_train(self, teacher_data, student_data):
        # sort the teacher and student input to avoid some bugs
        #！注意，这边的输入都是batch个数据，所以image_metas是多个图片的meta集合
        tnames = [meta["filename"] for meta in teacher_data["img_metas"]]
        snames = [meta["filename"] for meta in student_data["img_metas"]]
        tidx = [tnames.index(name) for name in snames]
        
        #start = time.time()
        #starttime = datetime.datetime.now() 
        with torch.no_grad():
            teacher_info = self.extract_teacher_info(
                teacher_data["img"][
                    torch.Tensor(tidx).to(teacher_data["img"].device).long()
                ],
                [teacher_data["img_metas"][idx] for idx in tidx],
                [teacher_data["proposals"][idx] for idx in tidx]
                if ("proposals" in teacher_data)
                and (teacher_data["proposals"] is not None)
                else None,
            )
        
        with torch.no_grad():
            teacher_info_down = self.extract_teacher_info_scale(
                teacher_data["img"][
                    torch.Tensor(tidx).to(teacher_data["img"].device).long()
                ],
                [teacher_data["img_metas"][idx] for idx in tidx],
                [teacher_data["proposals"][idx] for idx in tidx]
                if ("proposals" in teacher_data)
                and (teacher_data["proposals"] is not None)
                else None,
                mode = "down_wopad",
            )
        
        #student_info = self.extract_student_info(**student_data)
        #endtime = datetime.datetime.now() 
        #end = time.time()
        #print("cost_time:", endtime - starttime) #需要cost_time: 0:00:05.281560
        #print("time:", end - start)

        return self.compute_pseudo_label_loss(student_data["img"], student_data["img_metas"],  teacher_info, teacher_info_down)
        #return self.compute_pseudo_label_loss(student_data["img"], student_data["img_metas"],  teacher_info, teacher_info_down)

#!修改
    def compute_pseudo_label_loss(self, img, img_metas, teacher_info, teacher_info_down):
        
        x = self.student.extract_feat(img)

        student_transform_matrix = [
            torch.from_numpy(meta["transform_matrix"]).float().to(x[0][0].device)
            for meta in img_metas
        ]
        
        M = self._get_trans_mat(
            teacher_info["transform_matrix"], student_transform_matrix
        )

#！由于两个通道的数据增强不一致，需要将teacher的mask变换到student上，mask的变换----------------------#
        pseudo_masks = self._transform_mask(
            teacher_info["det_masks"],   #list(bitmapmasks)    3, 
            M,
            [meta["img_shape"] for meta in img_metas],
        )
        gt_labels = list(teacher_info["det_labels"])
        gt_scores =  list(teacher_info["det_scores"])
        gt_ious = list(teacher_info["det_ious"])

        pseudo_masks_down = self._transform_mask(
            teacher_info_down["det_masks"],   #list(bitmapmasks)    3, 
            M,
            [meta["img_shape"] for meta in img_metas],
        )
        gt_labels_down = list(teacher_info_down["det_labels"])
        gt_scores_down =  list(teacher_info_down["det_scores"])
        gt_ious_down = list(teacher_info_down["det_ious"])
        
        
        interval = 1000#----------------------------------------------------------------------------------#
        flag, count_record = isVisualbyCount_return_count(interval)

        # gt_masks and gt_semantic_seg are not padded when forming batch
        gt_masks_tensor = []
        gt_masks_tensor_down = []
        gt_masks_score = []
        gt_masks_score_down = []
        # batch_input_shape shoud be the same across images
        pad_H, pad_W = img_metas[0]['batch_input_shape']
        assign_H = pad_H // self.student.mask_assign_stride
        assign_W = pad_W // self.student.mask_assign_stride

#可视化下采样2倍的图片进行预测后的mask-------------#
        for i, gt_mask in enumerate(pseudo_masks_down):
            mask_tensor = gt_mask.to_tensor(torch.float, gt_labels[0].device)
            if gt_mask.width != pad_W or gt_mask.height != pad_H:
                pad_wh = (0, pad_W - gt_mask.width, 0, pad_H - gt_mask.height)
                mask_tensor = F.pad(mask_tensor, pad_wh, value=0)
            
            #if flag == 1:
                #self.maskVis(img[i], mask_tensor, gt_labels_down[i], count_record, img_metas[i], "pesudo_mask_down")                          
            
            if mask_tensor.shape[0] == 0:
                gt_masks_tensor_down.append(
                    mask_tensor.new_zeros(
                        (mask_tensor.size(0), assign_H, assign_W)))
                gt_masks_score_down.append(
                    mask_tensor.new_zeros(
                        (mask_tensor.size(0), assign_H, assign_W)))
            else:
                mask_tensor = F.interpolate(
                        mask_tensor[None], (assign_H, assign_W),
                        mode='bilinear',
                        align_corners=False)[0]
                mask_tensor = mask_tensor > 0.5
                mask_tensor = mask_tensor.float()
                nonzero_dim = mask_tensor.sum((1, 2)).float() > 0
                mask_tensor = mask_tensor[nonzero_dim]
                gt_masks_tensor_down.append(mask_tensor)

                gt_labels_down[i] = gt_labels_down[i][nonzero_dim]
                gt_scores_down[i] = gt_scores_down[i][nonzero_dim]
                gt_ious_down[i] = gt_ious_down[i][nonzero_dim]

                mask_score = teacher_info_down["det_mask_scores"][i][nonzero_dim]
                if mask_score.size(0) == 0:
                    gt_masks_score_down.append(mask_tensor.new_zeros((mask_score.size(0), assign_H, assign_W)))
                    continue
                mask_score = Transform2D.transform_mask_scores(mask_score, M[i], img_metas[i]["img_shape"])
                _, h, w = mask_score.shape
                if w != pad_W or h != pad_H:
                    pad_wh = (0, pad_W - gt_mask.width, 0, pad_H - gt_mask.height)
                    mask_score = F.pad(mask_score, pad_wh, value=0)
                gt_masks_score_down.append(F.interpolate(
                        mask_score[None], (assign_H, assign_W),
                        mode='bilinear',
                        align_corners=False)[0])

#可视化原始分辨率图片进行预测后的mask-------------#
        for i, gt_mask in enumerate(pseudo_masks):
            mask_tensor = gt_mask.to_tensor(torch.float, gt_labels[0].device)
            if gt_mask.width != pad_W or gt_mask.height != pad_H:
                pad_wh = (0, pad_W - gt_mask.width, 0, pad_H - gt_mask.height)
                mask_tensor = F.pad(mask_tensor, pad_wh, value=0)
            
            #if flag == 1:
                #self.maskVis(img[i], mask_tensor, gt_labels[i], count_record, img_metas[i], "pesudo_mask")                           
            
            
            if mask_tensor.shape[0] == 0:
                gt_masks_tensor.append(
                    mask_tensor.new_zeros(
                        (mask_tensor.size(0), assign_H, assign_W)))
                gt_masks_score.append(
                    mask_tensor.new_zeros(
                        (mask_tensor.size(0), assign_H, assign_W)))
            else:
                mask_tensor = F.interpolate(
                        mask_tensor[None], (assign_H, assign_W),
                        mode='bilinear',
                        align_corners=False)[0]
                mask_tensor = mask_tensor > 0.5
                mask_tensor = mask_tensor.float()
                nonzero_dim = mask_tensor.sum((1, 2)).float() > 0
                mask_tensor = mask_tensor[nonzero_dim]
                gt_masks_tensor.append(mask_tensor)

                gt_labels[i] = gt_labels[i][nonzero_dim]
                gt_scores[i] = gt_scores[i][nonzero_dim]
                gt_ious[i] = gt_ious[i][nonzero_dim]

                mask_score = teacher_info["det_mask_scores"][i][nonzero_dim]
                if mask_score.size(0) == 0:
                    gt_masks_score.append(mask_tensor.new_zeros((mask_score.size(0), assign_H, assign_W)))
                    continue
                mask_score = Transform2D.transform_mask_scores(mask_score, M[i], img_metas[i]["img_shape"])
                _, h, w = mask_score.shape
                if w != pad_W or h != pad_H:
                    pad_wh = (0, pad_W - gt_mask.width, 0, pad_H - gt_mask.height)
                    mask_score = F.pad(mask_score, pad_wh, value=0)
                gt_masks_score.append(F.interpolate(
                        mask_score[None], (assign_H, assign_W),
                        mode='bilinear',
                        align_corners=False)[0])
        '''
#可视化原分辨率的mask得分经过变换再二值化的mask-------------#
        #mask_scores = teacher_info["det_mask_scores"]
        for i in range(len(gt_masks_score)):
            mask_tensor = gt_masks_score[i]
            if mask_tensor.shape[0] == 0:
                continue
            mask_tensor = F.interpolate(mask_tensor[None], size=(pad_H, pad_W), mode='bilinear', align_corners=False).squeeze(0)
            #mask_tensor = Transform2D.transform_mask_scores(mask_tensor, M[i], img_metas[i]["batch_input_shape"])
            if flag == 1:   
                self.maskVis(img[i], mask_tensor, gt_labels[i], count_record, img_metas[i], "pesudo_mask_score")          


#可视化下采样两倍的mask得分经过变换再二值化的mask-------------#
        #mask_scores = teacher_info_scale["det_mask_scores"]
        for i in range(len(gt_masks_score_scale)):
            mask_tensor = gt_masks_score_scale[i]
            if mask_tensor.shape[0] == 0:
                continue
            mask_tensor = F.interpolate(mask_tensor[None], size=(pad_H, pad_W), mode='bilinear', align_corners=False).squeeze(0)
            #mask_tensor = Transform2D.transform_mask_scores(mask_tensor, M[i], img_metas[i]["batch_input_shape"])
            if flag == 1:  
                self.maskVis(img[i], mask_tensor, gt_labels_scale[i], count_record, img_metas[i], "pesudo_mask_score_scale")          
        '''    
                
        gt_masks = gt_masks_tensor
        gt_pred_scores = gt_masks_score

        gt_masks_down = gt_masks_tensor_down
        gt_pred_scores_down = gt_masks_score_down

        gt_masks_refine = []
        gt_pred_scores_refine = []
        gt_scores_refine =  []
        gt_ious_refine = []
        gt_labels_refine = []

        gt_masks_refine_lm = []
        gt_pred_scores_refine_lm = []
        gt_scores_refine_lm =  []
        gt_ious_refine_lm = []
        gt_labels_refine_lm = []

        #gt_masks, gt_labels, gt_scores, gt_ious, gt_masks_down, gt_labels_down, gt_scores_down, gt_ious_down = self.pseudo_label_rectify(gt_masks, gt_labels, gt_scores, gt_ious, gt_masks_down, gt_labels_down, gt_scores_down, gt_ious_down)
        for i in range(len(gt_masks)):
            fusion_mask_tensor = torch.cat((gt_masks_tensor[i], gt_masks_tensor_down[i]), dim = 0)
            fusion_mask_pred_scores = torch.cat((gt_pred_scores[i], gt_pred_scores_down[i]), dim = 0)
            fusion_labels = torch.cat((gt_labels[i], gt_labels_down[i]), dim = 0)
            fusion_cls_score = torch.cat((gt_scores[i], gt_scores_down[i]), dim = 0)
            fusion_mask_iou_score = torch.cat((gt_ious[i], gt_ious_down[i]), dim = 0)
            mix_score = fusion_mask_iou_score * fusion_cls_score
            flag_z = 0
            if fusion_mask_tensor.size(0) != 0:
                scores, labels, masks, keep_inds, s_a_d, s_b_d = mask_matrix_nms_three_steps_dythr(fusion_mask_tensor, fusion_mask_pred_scores, fusion_mask_iou_score, fusion_labels, fusion_cls_score, self.score_thresholds, is_hard = True)
                fusion_mask_pred_scores = fusion_mask_pred_scores[keep_inds]
                fusion_cls_score = fusion_cls_score[keep_inds]
                fusion_mask_iou_score = fusion_mask_iou_score[keep_inds]
                fusion_labels = fusion_labels[keep_inds]
                if masks.size(0) == 0:
                    flag_z = 1
                if flag_z != 1:
                    gt_masks_refine.append(masks)
                    mask_area = ((masks > 0.5).float()).sum((1, 2)).float()
                    #mask_area = masks.sum((1, 2)).float()
                    large_medium_indx = torch.where(mask_area >= 8 * img_metas[i]["scale_factor"][0] * 8 * img_metas[i]["scale_factor"][1])
                    gt_scores_refine.append(fusion_cls_score)
                    gt_ious_refine.append(fusion_mask_iou_score)
                    gt_labels_refine.append(fusion_labels)
                    
                    masks_lm = masks[large_medium_indx]
                    if masks_lm.size(0) != 0:
                        masks_lm = ((F.interpolate(masks_lm[None], size=(assign_H // 2, assign_W // 2), mode='bilinear', align_corners=False).squeeze(0)) > 0.5).float()
                        gt_masks_refine_lm.append(masks_lm)
                    else:
                        gt_masks_refine_lm.append(masks_lm.new_zeros((masks_lm.size(0), assign_H // 2, assign_W // 2)))
                    gt_scores_refine_lm.append(fusion_cls_score[large_medium_indx])
                    gt_ious_refine_lm.append(fusion_mask_iou_score[large_medium_indx])
                    gt_labels_refine_lm.append(fusion_labels[large_medium_indx])
                else:
                    gt_masks_refine.append(gt_masks_tensor[i])
                    gt_scores_refine.append(gt_scores[i])
                    gt_ious_refine.append(gt_ious[i])
                    gt_labels_refine.append(gt_labels[i])
                    
                    if gt_masks_tensor[i].size(0) != 0:
                        gt_masks_refine_lm.append(((F.interpolate(gt_masks_tensor[i][None], size=(assign_H // 2, assign_W // 2), mode='bilinear', align_corners=False).squeeze(0)) > 0.5).float())
                        gt_scores_refine_lm.append(gt_scores[i])
                        gt_ious_refine_lm.append(gt_ious[i])
                        gt_labels_refine_lm.append(gt_labels[i])
                    else:
                        gt_masks_refine_lm.append(((F.interpolate(gt_masks_tensor_down[i][None], size=(assign_H // 2, assign_W // 2), mode='bilinear', align_corners=False).squeeze(0)) > 0.5).float())
                        gt_scores_refine_lm.append(gt_scores_down[i])
                        gt_ious_refine_lm.append(gt_ious_down[i])
                        gt_labels_refine_lm.append(gt_labels_down[i])
            else:
                gt_masks_refine.append(fusion_mask_tensor)
                gt_scores_refine.append(fusion_cls_score)
                gt_ious_refine.append(fusion_mask_iou_score)
                gt_labels_refine.append(fusion_labels)

                gt_masks_refine_lm.append(fusion_mask_tensor.new_zeros((fusion_mask_tensor.size(0), assign_H // 2, assign_W // 2)))
                gt_scores_refine_lm.append(fusion_cls_score)
                gt_ious_refine_lm.append(fusion_mask_iou_score)
                gt_labels_refine_lm.append(fusion_labels)
        #gt_masks_r, gt_labels_r, gt_scores_r, gt_ious_r, gt_pred_scores_r = self.pseudo_mask_fusion_refine(gt_masks, gt_pred_scores, gt_labels, gt_scores, gt_ious, gt_masks_scale, gt_pred_scores_scale, gt_labels_scale, gt_scores_scale, gt_ious_scale)

        if flag == 1:
            self.mask2bbox_Vis(img, gt_masks, gt_labels, count_record, img_metas, "bbox", gt_scores)
            self.mask2bbox_Vis(img, gt_masks_down, gt_labels_down, count_record, img_metas, "bbox_down", gt_scores_down)
            self.mask2bbox_Vis(img, gt_masks_refine, gt_labels_refine, count_record, img_metas, "bbox_refine", gt_scores_refine, mode="soft")
            self.mask2bbox_Vis(img, gt_pred_scores, gt_labels, count_record, img_metas, "scores", gt_scores, mode="soft")
            self.mask2bbox_Vis(img, gt_pred_scores_down, gt_labels_down, count_record, img_metas, "scores_down", gt_scores_down, mode="soft")
            '''
            for i in range(len(gt_pred_scores_r)):
                mask_tensor = gt_pred_scores_r[i]
                if mask_tensor.size(0) == 0:
                    continue
                mask_tensor = F.interpolate(
                        mask_tensor[None], size = (pad_H, pad_W),
                        mode='bilinear',
                        align_corners=False)[0]
                self.maskVis(img[i], mask_tensor, gt_labels_r[i], count_record, img_metas[i], "pesudo_mask_scores_refinement")
                mask_tensor = gt_masks_r[i]
                if mask_tensor.size(0) == 0:
                    continue
                mask_tensor = F.interpolate(
                        mask_tensor[None], size = (pad_H, pad_W),
                        mode='bilinear',
                        align_corners=False)[0]
                self.maskVis(img[i], mask_tensor, gt_labels_r[i], count_record, img_metas[i], "pesudo_mask_refinement") 
            '''
        #gt_mask_refine, gt_scores_refine, gt_ious_refine = self.refine_pseudo_mask(gt_masks, gt_scores, gt_ious, gt_masks_scale, gt_scores_scale, gt_ious_scale, flag)
        
        rpn_results = self.student.rpn_head.forward_train(x, img_metas, gt_masks_refine,
                                                  gt_labels_refine, gt_scores_refine, gt_ious_refine)
        
        scale = 2
        img_down = self.test_scale_mask(img, scale, "down_wopad", img_metas)
        x_down = self.student.extract_feat(img_down)
        for i in range(len(gt_masks_refine_lm)):
            _, h, w = gt_masks_refine_lm[i].shape
            pad_h = x_down[0].shape[-2] - h
            pad_w = x_down[0].shape[-1] - w
            pad_wh = (0, pad_w, 0, pad_h)
            gt_masks_refine_lm[i] = F.pad(gt_masks_refine_lm[i], pad_wh, value=0)
        rpn_results_down = self.student.rpn_head.forward_train(x_down, img_metas, gt_masks_refine_lm,
                                                  gt_labels_refine_lm, gt_scores_refine_lm, gt_ious_refine_lm, is_scale = "_down", mode = "down")
        (rpn_losses, proposal_feats, x_feats, mask_preds,
         cls_scores) = rpn_results
        (rpn_losses_down, proposal_feats_down, x_feats_down, mask_preds_down,
            cls_scores_down) = rpn_results_down
        
        loss_rpn_seg = rpn_losses['loss_rpn_seg']
        rpn_losses['loss_rpn_seg'] = loss_rpn_seg * 0

        loss_rpn_seg_down = rpn_losses_down['loss_rpn_seg_down']
        rpn_losses_down['loss_rpn_seg_down'] = loss_rpn_seg_down * 0
        rpn_losses_down = weighted_loss(rpn_losses_down, weight=0.5)
#------------------------------------#
        #print("mask_preds.shape:", mask_preds.shape)
        losses = self.student.roi_head.forward_train(
            x_feats,
            proposal_feats,
            mask_preds,
            cls_scores,
            img_metas,
            gt_masks_refine,
            gt_labels_refine,
            gt_scores_refine,
            gt_ious_refine,
            imgs_whwh=None)
        iou_loss = losses["s0_loss_iou"]
        losses["s0_loss_iou"] = iou_loss * 0
        iou_loss = losses["s1_loss_iou"]
        losses["s1_loss_iou"] = iou_loss * 0        
        iou_loss = losses["s2_loss_iou"]
        losses["s2_loss_iou"] = iou_loss * 0

        
        losses_down = weighted_loss(self.student.roi_head.forward_train(
            x_feats_down,
            proposal_feats_down,
            mask_preds_down,
            cls_scores_down,
            img_metas,
            gt_masks_refine_lm,
            gt_labels_refine_lm,
            gt_scores_refine_lm,
            gt_ious_refine_lm,
            is_scale='_down',
            imgs_whwh=None), weight=0.5)
        iou_loss_down = losses_down["s0_loss_iou_down"]
        losses_down["s0_loss_iou_down"] = iou_loss_down * 0
        iou_loss_down = losses_down["s1_loss_iou_down"]
        losses_down["s1_loss_iou_down"] = iou_loss_down * 0        
        iou_loss_down = losses_down["s2_loss_iou_down"]
        losses_down["s2_loss_iou_down"] = iou_loss_down * 0
        
        
        losses.update(rpn_losses)
        losses.update(losses_down)
        losses.update(rpn_losses_down)
        losses["gmm_thr"] = torch.tensor(teacher_info["gmm_thr"], device=img[0].device)
        losses["gmm_thr_down"] = torch.tensor(teacher_info_down["gmm_thr_down"], device=img[0].device)
        return losses
   

    @force_fp32(apply_to=["bboxes", "trans_mat"])
    def _transform_bbox(self, bboxes, trans_mat, max_shape):
        bboxes = Transform2D.transform_bboxes(bboxes, trans_mat, max_shape)
        return bboxes
    
    
#！加一个函数，类似于_transform_bbox，用于mask的变换
    #@force_fp32(apply_to=["masks", "trans_mat"])
    
    def _transform_mask(self, masks, trans_mat, max_shape):
        masks = Transform2D.transform_masks(masks, trans_mat, max_shape)
        return masks

#-------------------------------------------------#

    @force_fp32(apply_to=["a", "b"])
    def _get_trans_mat(self, a, b):
        return [bt @ at.inverse() for bt, at in zip(b, a)]


#！ 修改
#!!!!!!!!!
    def extract_teacher_info(self, img, img_metas, proposals=None, **kwargs):
        teacher_info = {}
        feat = self.teacher.extract_feat(img)
        teacher_info["backbone_feature"] = feat
        #不需要保存teacher的proposal
        
        
        #start2 = time.time()
        rpn_outs = self.teacher.rpn_head.simple_test_rpn(feat, img_metas)
        (proposal_feats, x_feats, mask_preds, cls_scores, seg_preds) = rpn_outs
        #roi_outs = self.teacher.roi_head.simple_test(x_feats, proposal_feats, mask_preds, cls_scores, img_metas)
        #num_roi_outs = len(roi_outs)
        #seg_results, label_results, score_results = self.teacher.roi_head.teacher_test(x_feats, proposal_feats, mask_preds, cls_scores, img_metas)
#!这边有分支，可以选择带iou的，也可以选择不带iou的
#--------------------------------------------#
        #seg_results, label_results, score_results, _ = self.teacher.roi_head.teacher_test(x_feats, proposal_feats, mask_preds, cls_scores, img_metas)
        seg_results, mask_scores, label_results, score_results, iou_results = self.teacher.roi_head.teacher_test(x_feats, proposal_feats, mask_preds, cls_scores, img_metas)
        #end2 = time.time()
        #print("test_cost:", end2 - start2)
        
        thrs = []
        for i, proposals in enumerate(score_results):
            dynamic_ratio = self.train_cfg.dynamic_ratio
            scores = proposals
            judge_nan = torch.isnan(scores)
            if judge_nan.float().sum() > 0:
                thrs.append(1)
            else:
                # num_gt = int(scores.sum() + 0.5)
                num_gt = int(scores.sum() * dynamic_ratio + 0.5)
                num_gt = min(num_gt, len(scores) - 1)
                thrs.append(scores[num_gt] - 1e-5)

        #cls_thr = self.train_cfg.pseudo_label_initial_score_thr
        #iou_thr = self.train_cfg.pseudo_label_iou_thr

        det_masks, det_mask_scores, det_labels, det_scores, det_ious = list(
            zip(
                *[
                    filter_invalid_stage1(
                        mask=seg_result,
                        mask_score=mask_score,
                        label=label_result,
                        score=score_result,
                        iou=iou_result,
                        thr=thr,
                    )
                    for seg_result, mask_score, label_result, score_result, iou_result, thr in zip(
                        seg_results, mask_scores, label_results, score_results, iou_results, thrs
                    )
                ]
            )
        )

        scores = torch.cat(det_scores)
        labels = torch.cat(det_labels)
        thrs = torch.zeros_like(scores)

        for label in torch.unique(labels):
            label = int(label)
            scores_add = (scores[labels == label])
            num_buffers = len(self.scores[label])
            scores_new= torch.cat([scores_add.float(), self.scores[label].float()])[:num_buffers]
            self.scores[label] = scores_new
            thr = self.gmm_policy(
                scores_new[scores_new > 0],
                given_gt_thr=self.train_cfg.get('given_gt_thr', 0),
                policy=self.train_cfg.get('policy', 'high'))
            #if thr > 0.35:
                #thr = 0.35
            thrs[labels == label] = thr
            self.score_thresholds[label] = thr
        mean_thr = thrs.mean()
        if len(thrs) == 0:
            mean_thr.fill_(0)
        mean_thr = float(mean_thr)
        teacher_info["gmm_thr"] = mean_thr
        thrs = torch.split(thrs, [len(p) for p in det_labels])

        det_masks, det_mask_scores, det_labels, det_scores, det_ious = list( 
            zip(
                *[
                    filter_invalid_stage1(
                        mask=seg_result,
                        mask_score=mask_score,
                        label=label_result,
                        score=score_result,
                        iou=iou_result,
                        thr=thr,
                    )
                    for seg_result, mask_score, label_result, score_result, iou_result, thr in zip(
                        det_masks, det_mask_scores, det_labels, det_scores, det_ious, thrs
                    )
                ]
            )
        )
        
        
        teacher_info["det_masks"] = det_masks
        teacher_info["det_mask_scores"] = det_mask_scores
        teacher_info["det_labels"] = det_labels
        teacher_info["det_scores"] = det_scores
        teacher_info["det_ious"] = det_ious
        teacher_info["transform_matrix"] = [
            torch.from_numpy(meta["transform_matrix"]).float().to(feat[0][0].device)
            for meta in img_metas
        ]
        teacher_info["img_metas"] = img_metas
        return teacher_info


    @staticmethod
    def aug_box(boxes, times=1, frac=0.06):
        def _aug_single(box):
            # random translate
            # TODO: random flip or something
            box_scale = box[:, 2:4] - box[:, :2]
            box_scale = (
                box_scale.clamp(min=1)[:, None, :].expand(-1, 2, 2).reshape(-1, 4)
            )
            aug_scale = box_scale * frac  # [n,4]

            offset = (
                torch.randn(times, box.shape[0], 4, device=box.device)
                * aug_scale[None, ...]
            )
            new_box = box.clone()[None, ...].expand(times, box.shape[0], -1)
            return torch.cat(
                [new_box[:, :, :4].clone() + offset, new_box[:, :, 4:]], dim=-1
            )

        return [_aug_single(box) for box in boxes]

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        if not any(["student" in key or "teacher" in key for key in state_dict.keys()]):
            keys = list(state_dict.keys())
            state_dict.update({"teacher." + k: state_dict[k] for k in keys})
            state_dict.update({"student." + k: state_dict[k] for k in keys})
            for k in keys:
                state_dict.pop(k)

        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def extract_teacher_info_scale(self, img, img_metas, proposals=None, mode=None, **kwargs):
        teacher_info = {}
        _, _, h, w = img.shape
        scale = 2
        img_scale = self.test_scale_mask(img, scale, mode, img_metas)

        feat = self.teacher.extract_feat(img_scale)
        teacher_info["backbone_feature"] = feat
        
        rpn_outs = self.teacher.rpn_head.simple_test_rpn(feat, img_metas)
        (proposal_feats, x_feats, mask_preds, cls_scores, seg_preds) = rpn_outs
        #roi_outs = self.teacher.roi_head.simple_test(x_feats, proposal_feats, mask_preds, cls_scores, img_metas)
        #num_roi_outs = len(roi_outs)
        #seg_results, label_results, score_results = self.teacher.roi_head.teacher_test(x_feats, proposal_feats, mask_preds, cls_scores, img_metas)
#!这边有分支，可以选择带iou的，也可以选择不带iou的
#--------------------------------------------#
        #seg_results, label_results, score_results, _ = self.teacher.roi_head.teacher_test(x_feats, proposal_feats, mask_preds, cls_scores, img_metas)
        seg_results, mask_scores, label_results, score_results, iou_results = self.teacher.roi_head.teacher_test_scale(x_feats, proposal_feats, mask_preds, cls_scores, img_metas, scale = scale, mode = mode)
        #end2 = time.time()
        #print("test_cost:", end2 - start2)
        
        thrs = []
        for i, proposals in enumerate(score_results):
            dynamic_ratio = self.train_cfg.dynamic_ratio
            scores = proposals
            judge_nan = torch.isnan(scores)
            if judge_nan.float().sum() > 0:
                thrs.append(1)
            else:
                # num_gt = int(scores.sum() + 0.5)
                num_gt = int(scores.sum() * dynamic_ratio + 0.5)
                num_gt = min(num_gt, len(scores) - 1)
                thrs.append(scores[num_gt] - 1e-5)

        #cls_thr = self.train_cfg.pseudo_label_initial_score_thr
        #iou_thr = self.train_cfg.pseudo_label_iou_thr

        det_masks, det_mask_scores, det_labels, det_scores, det_ious = list(
            zip(
                *[
                    filter_invalid_stage1(
                        mask=seg_result,
                        mask_score=mask_score,
                        label=label_result,
                        score=score_result,
                        iou=iou_result,
                        thr=thr,
                    )
                    for seg_result, mask_score, label_result, score_result, iou_result, thr in zip(
                        seg_results, mask_scores, label_results, score_results, iou_results, thrs
                    )
                ]
            )
        )

        scores = torch.cat(det_scores)
        labels = torch.cat(det_labels)
        thrs = torch.zeros_like(scores)

        for label in torch.unique(labels):
            label = int(label)
            scores_add = (scores[labels == label])
            num_buffers = len(self.scores[label])
            scores_new= torch.cat([scores_add.float(), self.scores[label].float()])[:num_buffers]
            self.scores[label] = scores_new
            thr = self.gmm_policy(
                scores_new[scores_new > 0],
                given_gt_thr=self.train_cfg.get('given_gt_thr', 0),
                policy=self.train_cfg.get('policy', 'high'))
            #if thr > 0.35:
                #thr = 0.35
            thrs[labels == label] = thr
            self.score_thresholds[label] = thr
        mean_thr = thrs.mean()
        if len(thrs) == 0:
            mean_thr.fill_(0)
        mean_thr = float(mean_thr)
        teacher_info["gmm_thr_down"] = mean_thr
        thrs = torch.split(thrs, [len(p) for p in det_labels])

        det_masks, det_mask_scores, det_labels, det_scores, det_ious = list(
            zip(
                *[
                    filter_invalid_stage1(
                        mask=seg_result,
                        mask_score=mask_score,
                        label=label_result,
                        score=score_result,
                        iou=iou_result,
                        thr=thr,
                    )
                    for seg_result, mask_score, label_result, score_result, iou_result, thr in zip(
                        det_masks, det_mask_scores, det_labels, det_scores, det_ious, thrs
                    )
                ]
            )
        )
        
        teacher_info["det_masks"] = det_masks
        teacher_info["det_mask_scores"] = det_mask_scores
        teacher_info["det_labels"] = det_labels
        teacher_info["det_scores"] = det_scores
        teacher_info["det_ious"] = det_ious
        teacher_info["transform_matrix"] = [
            torch.from_numpy(meta["transform_matrix"]).float().to(feat[0][0].device)
            for meta in img_metas
        ]
        teacher_info["img_metas"] = img_metas
        return teacher_info

    #def refine_pseudo_mask(self, img, img_metas, proposals=None, **kwargs):
    
    def test_scale_mask(self, img, scale, mode, img_metas):
        if scale == 1:
            return img
        _, _, h, w = img.shape
        if mode == "down_wpad":
            img_downsample = F.interpolate(img, (h//scale, w//scale), mode='bilinear', align_corners=False)
            pad_wh = (0, w - w//scale, 0, h - h//scale)
            img_downsample = F.pad(img_downsample, pad_wh, value=0)
            return img_downsample
        elif mode == "down_wopad":
            img_downsample = F.interpolate(img, (h//scale, w//scale), mode='bilinear', align_corners=False)
            _, _, h, w = img_downsample.shape
            img_metas[0]["downsample_shape"] = (h // 4, w // 4)
            if w % 32 == 0:
                pad_w = 0
            else:
                pad_w = 32 * (w//32 + 1) - w
            if h % 32 == 0:
                pad_h = 0
            else:
                pad_h = 32 * (h//32 + 1) - h
            pad_wh = (0, pad_w, 0, pad_h)
            img_downsample = F.pad(img_downsample, pad_wh, value=0)
            return img_downsample
        elif mode == "up":
            img_up = F.interpolate(img, scale_factor = scale, mode='bilinear', align_corners=False)
            return img_up

    def mask2bbox_Vis(self, img, gt_masks, gt_labels, count_record, img_metas, workdir, det_scores, mode=None):
        pad_H, pad_W = img_metas[0]['batch_input_shape']
        assign_H = pad_H // self.student.mask_assign_stride
        assign_W = pad_W // self.student.mask_assign_stride
        for i in range(len(gt_masks)):
            bbox_list = []
            box_mask = None
            if mode=="soft":
                mask_tensor = (gt_masks[i] > 0.5).float()
            else:
                mask_tensor = gt_masks[i]
            if mask_tensor.shape[0] != 0:
                mask_tensor = F.interpolate(
                        mask_tensor[None], size = (pad_H, pad_W),
                        mode='bilinear',
                        align_corners=False)[0]
                for id in range(mask_tensor.shape[0]):
                    box_single_mask = None
                    mask_temp = mask_tensor[id]
                    coor = torch.nonzero(mask_temp)
                    if coor.shape[0] == 0:
                        xmin = 0
                        xmax = 0
                        ymin = 0
                        ymax = 0
                        box_single_mask = torch.zeros([assign_H, assign_W], device = img[0][0].device)
                    else:
                        xmin = torch.min(coor[:, 1]).item()
                        xmax = torch.max(coor[:, 1]).item()

                        ymin = torch.min(coor[:, 0]).item()
                        ymax = torch.max(coor[:, 0]).item()

                    bbox_list.append([xmin, ymin, xmax, ymax, det_scores[i][id].item()])
                    if box_single_mask == None:
                        box_single_mask = torch.zeros([assign_H, assign_W], device = img[0][0].device)
                        box_single_mask[ymin // self.student.mask_assign_stride: ymax // self.student.mask_assign_stride + 1, xmin // self.student.mask_assign_stride: xmax // self.student.mask_assign_stride + 1] = 1

                    if box_mask == None:
                        box_mask = box_single_mask.unsqueeze(0)
                    else:
                        box_mask = torch.cat((box_mask, box_single_mask.unsqueeze(0)), 0)
            bbox_tensor = torch.tensor(bbox_list)

            image_vis = img[i].cpu().detach()
            label_vis = gt_labels[i].cpu().detach()
            bbox_vis = bbox_tensor
            mask_vis = mask_tensor.cpu().detach()
            mask_vis = mask_vis > 0.5
            if  mask_tensor.shape[0] > 0:
                log_image_with_masks_without_box(
                    workdir,
                    image_vis,
                    bbox_vis,
                    mask_vis,
                    bbox_tag=workdir,
                    labels=label_vis,
                    class_names=self.CLASSES,
                    interval=1,
                    img_norm_cfg=img_metas[i]["img_norm_cfg"],
                    filename= str(count_record) + "_" + img_metas[i]["filename"].split("/")[-1]
                )
    
    def rm_all_zero_mask(self, mask_tensor):
        mask_tensor_collect = []
        for i in range(len(gt_masks)):
            mask_tensor = gt_masks[i]
            if mask_tensor.shape[0] != 0:
                mask_tensor = mask_tensor[mask_tensor.sum((1, 2)).float > 0]
                mask_tensor_collect.append(mask_tensor)
            else:
                mask_tensor_collect.append(mask_tensor)


    def maskVis(self, img, mask_tensor, gt_labels, count_record, img_metas, workdir):
        image_vis = img.cpu().detach()
        mask_vis = mask_tensor.cpu().detach()
        mask_vis = mask_vis > 0.5
        label_vis = gt_labels.cpu().detach()

        if  mask_tensor.shape[0] > 0:
            log_image_with_masks_without_box(
                workdir,
                image_vis,
                None,
                mask_vis,
                bbox_tag=workdir,
                labels=label_vis,
                class_names=self.CLASSES,
                interval=1,
                img_norm_cfg=img_metas["img_norm_cfg"],
                filename= str(count_record) + "_" + img_metas["filename"].split("/")[-1]
            )
    
    def gmm_policy(self, scores, given_gt_thr=0.35, policy='high'):
        """The policy of choosing pseudo label.

        The previous GMM-B policy is used as default.
        1. Use the predicted bbox to fit a GMM with 2 center.
        2. Find the predicted bbox belonging to the positive
            cluster with highest GMM probability.
        3. Take the class score of the finded bbox as gt_thr.

        Args:
            scores (nd.array): The scores.

        Returns:
            float: Found gt_thr.

        """
        if len(scores) < 4:
            return given_gt_thr
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        if len(scores.shape) == 1:
            scores = scores[:, np.newaxis]
        means_init = [[np.min(scores)], [np.max(scores)]]
        weights_init = [1 / 2, 1 / 2]
        precisions_init = [[[1.0]], [[1.0]]]
        gmm = skm.GaussianMixture(
            2,
            weights_init=weights_init,
            means_init=means_init,
            precisions_init=precisions_init)
        gmm.fit(scores)
        gmm_assignment = gmm.predict(scores)
        gmm_scores = gmm.score_samples(scores)
        assert policy in ['middle', 'high']
        if policy == 'high':
            if (gmm_assignment == 1).any():
                gmm_scores[gmm_assignment == 0] = -np.inf
                indx = np.argmax(gmm_scores, axis=0)
                pos_indx = (gmm_assignment == 1) & (
                    scores >= scores[indx]).squeeze()
                pos_thr = float(scores[pos_indx].min())
                # pos_thr = max(given_gt_thr, pos_thr)
            else:
                pos_thr = given_gt_thr
        elif policy == 'middle':
            if (gmm_assignment == 1).any():
                pos_thr = float(scores[gmm_assignment == 1].min())
                # pos_thr = max(given_gt_thr, pos_thr)
            else:
                pos_thr = given_gt_thr

        return pos_thr