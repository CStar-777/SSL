import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule, force_fp32, auto_fp16
from mmcv.cnn import (ConvModule, bias_init_with_prob, normal_init)
from mmdet.core import build_assigner, build_sampler, multi_apply, reduce_mean
from mmdet.models.builder import HEADS, build_loss, build_neck
from mmdet.models.losses import accuracy
from mmdet.utils import get_root_logger


@HEADS.register_module()
class ConvKernelHead(BaseModule):

    def __init__(self,
                 num_proposals=100,
                 in_channels=256,
                 out_channels=256,
                 num_heads=8,
                 num_cls_fcs=1,
                 num_seg_convs=1,
                 num_loc_convs=1,
                 att_dropout=False,
                 localization_fpn=None,
                 conv_kernel_size=1,
                 norm_cfg=dict(type='GN', num_groups=32),
                 semantic_fpn=True,
                 train_cfg=None,
                 num_classes=80,
                 xavier_init_kernel=False,
                 kernel_init_std=0.01,
                 use_binary=False,
                 proposal_feats_with_obj=False,
                 loss_mask=None,
                 loss_mask_1=None,
                 loss_seg=None,
                 loss_cls=None,
                 loss_dice=None,
                 loss_rank=None,
                 loss_levelset=None,
                 loss_pairwise=None,
                 feat_downsample_stride=1,
                 feat_refine_stride=1,
                 feat_refine=True,
                 with_embed=False,
                 feat_embed_only=False,
                 conv_normal_init=False,
                 mask_out_stride=4,
                 hard_target=False,
                 num_thing_classes=80,
                 num_stuff_classes=53,
                 mask_assign_stride=4,
                 ignore_label=255,
                 thing_label_in_seg=0,
                 cat_stuff_mask=False,
                 init_cfg=None,
                 **kwargs):
        super(ConvKernelHead, self).__init__(init_cfg)
        self.num_proposals = num_proposals
        self.num_cls_fcs = num_cls_fcs
        self.train_cfg = train_cfg
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_classes = num_classes
        self.proposal_feats_with_obj = proposal_feats_with_obj
        self.sampling = False
        self.localization_fpn = build_neck(localization_fpn)
        self.semantic_fpn = semantic_fpn
        self.norm_cfg = norm_cfg
        self.num_heads = num_heads
        self.att_dropout = att_dropout
        self.mask_out_stride = mask_out_stride
        self.hard_target = hard_target
        self.conv_kernel_size = conv_kernel_size
        self.xavier_init_kernel = xavier_init_kernel
        self.kernel_init_std = kernel_init_std
        self.feat_downsample_stride = feat_downsample_stride
        self.feat_refine_stride = feat_refine_stride
        self.conv_normal_init = conv_normal_init
        self.feat_refine = feat_refine
        self.with_embed = with_embed
        self.feat_embed_only = feat_embed_only
        self.num_loc_convs = num_loc_convs
        self.num_seg_convs = num_seg_convs
        self.use_binary = use_binary
        self.num_thing_classes = num_thing_classes
        self.num_stuff_classes = num_stuff_classes
        self.mask_assign_stride = mask_assign_stride
        self.ignore_label = ignore_label
        self.thing_label_in_seg = thing_label_in_seg
        self.cat_stuff_mask = cat_stuff_mask
        self.fp16_enabled = False
        
        if loss_mask is not None:
            self.loss_mask = build_loss(loss_mask)
        else:
            self.loss_mask = loss_mask
        
        if loss_levelset is not None:
            self.loss_levelset = build_loss(loss_levelset)
        else:
            self.loss_levelset = loss_levelset
        
        if loss_pairwise is not None:
            self.loss_pairwise = build_loss(loss_pairwise)
        else:
            self.loss_pairwise = loss_pairwise

        if loss_mask_1 is not None:
            self.loss_mask_1 = build_loss(loss_mask_1)
        else:
            self.loss_mask_1 = loss_mask_1            
            
        if loss_dice is not None:
            self.loss_dice = build_loss(loss_dice)
        else:
            self.loss_dice = loss_dice

        if loss_seg is not None:
            self.loss_seg = build_loss(loss_seg)
        else:
            self.loss_seg = loss_seg
        if loss_cls is not None:
            self.loss_cls = build_loss(loss_cls)
        else:
            self.loss_cls = loss_cls

        if loss_rank is not None:
            self.loss_rank = build_loss(loss_rank)
        else:
            self.loss_rank = loss_rank

        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            # use PseudoSampler when sampling is False
            if self.sampling and hasattr(self.train_cfg, 'sampler'):
                sampler_cfg = self.train_cfg.sampler
            else:
                sampler_cfg = dict(type='MaskPseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self._init_layers()
        
#-------------------#
        
        if self.init_cfg is None:
            self._init_weights()
        

    def _init_layers(self):
        """Initialize a sparse set of proposal boxes and proposal features."""
        self.init_kernels = nn.Conv2d(
            self.out_channels,
            self.num_proposals,
            self.conv_kernel_size,
            padding=int(self.conv_kernel_size // 2),
            bias=False)

        #self.feat_conv = nn.Conv2d(self.out_channels, 3, 1)

        if self.semantic_fpn:
            if self.loss_seg.use_sigmoid:
                self.conv_seg = nn.Conv2d(self.out_channels, self.num_classes,
                                          1)
            else:
                self.conv_seg = nn.Conv2d(self.out_channels,
                                          self.num_classes + 1, 1)

        if self.feat_downsample_stride > 1 and self.feat_refine:
            self.ins_downsample = ConvModule(
                self.in_channels,
                self.out_channels,
                3,
                stride=self.feat_refine_stride,
                padding=1,
                norm_cfg=self.norm_cfg)
            self.seg_downsample = ConvModule(
                self.in_channels,
                self.out_channels,
                3,
                stride=self.feat_refine_stride,
                padding=1,
                norm_cfg=self.norm_cfg)

        self.loc_convs = nn.ModuleList()
        for i in range(self.num_loc_convs):
            self.loc_convs.append(
                ConvModule(
                    self.in_channels,
                    self.out_channels,
                    1,
                    norm_cfg=self.norm_cfg))

        self.seg_convs = nn.ModuleList()
        for i in range(self.num_seg_convs):
            self.seg_convs.append(
                ConvModule(
                    self.in_channels,
                    self.out_channels,
                    1,
                    norm_cfg=self.norm_cfg))
    
    def _init_weights(self):
        self.localization_fpn._init_weights()
       # normal_init(self.feat_conv, mean=0, std=0.01)
        #print("使用了自定义权重初始化")
        if self.feat_downsample_stride > 1 and self.conv_normal_init:
            logger = get_root_logger()
            logger.info('Initialize convs in KPN head by normal std 0.01')
            for conv in [self.loc_convs, self.seg_convs]:
                for m in conv.modules():
                    if isinstance(m, nn.Conv2d):
                        normal_init(m, std=0.01)

        if self.semantic_fpn:
            bias_seg = bias_init_with_prob(0.01)
            if self.loss_seg.use_sigmoid:
                normal_init(self.conv_seg, std=0.01, bias=bias_seg)
            else:
                normal_init(self.conv_seg, mean=0, std=0.01)
        if self.xavier_init_kernel:
            logger = get_root_logger()
            logger.info('Initialize kernels by xavier uniform')
            nn.init.xavier_uniform_(self.init_kernels.weight)
        else:
            logger = get_root_logger()
            logger.info(
                f'Initialize kernels by normal std: {self.kernel_init_std}')
            normal_init(self.init_kernels, mean=0, std=self.kernel_init_std)
    
    
    def _decode_init_proposals(self, img, img_metas):
        num_imgs = len(img_metas)

        localization_feats = self.localization_fpn(img)
        if isinstance(localization_feats, list):
            loc_feats = localization_feats[0]
        else:
            loc_feats = localization_feats
        for conv in self.loc_convs:
            loc_feats = conv(loc_feats)
        if self.feat_downsample_stride > 1 and self.feat_refine:
            loc_feats = self.ins_downsample(loc_feats)
        mask_preds = self.init_kernels(loc_feats)

        if self.semantic_fpn:
            if isinstance(localization_feats, list):
                semantic_feats = localization_feats[1]
            else:
                semantic_feats = localization_feats
            for conv in self.seg_convs:
                semantic_feats = conv(semantic_feats)
            if self.feat_downsample_stride > 1 and self.feat_refine:
                semantic_feats = self.seg_downsample(semantic_feats)
        else:
            semantic_feats = None

        if semantic_feats is not None:
            seg_preds = self.conv_seg(semantic_feats)
        else:
            seg_preds = None

        proposal_feats = self.init_kernels.weight.clone()
        proposal_feats = proposal_feats[None].expand(num_imgs,
                                                     *proposal_feats.size())

        if semantic_feats is not None:
            x_feats = semantic_feats + loc_feats
        else:
            x_feats = loc_feats

        if self.proposal_feats_with_obj:
            sigmoid_masks = mask_preds.sigmoid()
            nonzero_inds = sigmoid_masks > 0.5
            if self.use_binary:
                sigmoid_masks = nonzero_inds.float()
            else:
                sigmoid_masks = nonzero_inds.float() * sigmoid_masks
            obj_feats = torch.einsum('bnhw,bchw->bnc', sigmoid_masks, x_feats)

        cls_scores = None

        if self.proposal_feats_with_obj:
            proposal_feats = proposal_feats + obj_feats.view(
                num_imgs, self.num_proposals, self.out_channels, 1, 1)

        if self.cat_stuff_mask and not self.training:
            mask_preds = torch.cat(
                [mask_preds, seg_preds[:, self.num_thing_classes:]], dim=1)
            stuff_kernels = self.conv_seg.weight[self.
                                                 num_thing_classes:].clone()
            stuff_kernels = stuff_kernels[None].expand(num_imgs,
                                                       *stuff_kernels.size())
            proposal_feats = torch.cat([proposal_feats, stuff_kernels], dim=1)

        return proposal_feats, x_feats, mask_preds, cls_scores, seg_preds

    @force_fp32(apply_to=('img', 'gt_masks')) 
    def forward_train(self,
                      img,
                      img_metas,
                      gt_masks,
                      gt_labels,
                      gt_scores=None,
                      gt_ious=None,
                      gt_sem_seg=None,
                      gt_sem_cls=None):
        """Forward function in training stage."""
        num_imgs = len(img_metas)
        results = self._decode_init_proposals(img, img_metas)
        (proposal_feats, x_feats, mask_preds, cls_scores, seg_preds) = results

#----------------------------------------#        
        #print("mask_preds.dtype", mask_preds.dtype)  #torch.float16
        #print("seg_preds.shape", seg_preds.dtype, seg_preds.shape) #torch.Size([1, 80, 152, 100])
        #print("+"*100)
        #print("gt_masks[0].dtype", gt_masks[0].dtype) #torch.float16
        
        if self.feat_downsample_stride > 1:
            scaled_mask_preds = F.interpolate(
                mask_preds,
                scale_factor=self.feat_downsample_stride,
                mode='bilinear',
                align_corners=False)
            if seg_preds is not None:
                scaled_seg_preds = F.interpolate(
                    seg_preds,
                    scale_factor=self.feat_downsample_stride,
                    mode='bilinear',
                    align_corners=False)
        else:
            scaled_mask_preds = mask_preds
            scaled_seg_preds = seg_preds

        if self.hard_target:
            gt_masks = [x.bool().float() for x in gt_masks]
        else:
            gt_masks = gt_masks
            
        
#------------------------------#      
        """
        print("scaled_mask_preds.shape:", scaled_mask_preds.shape)   #torch.Size([1, 100, 240, 320])
        print("gt_masks.shape:", len(gt_masks), gt_masks[0].shape)   #1 torch.Size([5, 240, 320])
        print("gt_labels.shape", len(gt_labels), gt_labels[0].shape)
        print("gt_labels:", gt_labels)  
        """
        #print("scaled_mask_preds.shape:", scaled_mask_preds.shape)  
        #print("gt_labels.shape", len(gt_labels), gt_labels[0].shape)  #3 torch.Size([3]) 同mask_pred
        #print("scaled_mask_preds.dtype:", scaled_mask_preds.dtype) #torch.float16
        #print("gt_masks.dtype:", gt_masks[0].dtype)  #torch.float32
        #scaled_mask_preds = scaled_mask_preds.float()
        #print("scaled_mask_preds.dtype:", scaled_mask_preds.dtype)
        #print("gt_sem_seg.shape:", len(gt_sem_seg), gt_sem_seg[0].shape) #None
        #print("gt_sem_cls.shape:", len(gt_sem_cls), gt_sem_cls[0].shape) #None
        flag = 1
        if img_metas[0]["tag"] == "sup":
            flag = 1
        else:
            flag = 0
        
    
    
        sampling_results = []
        if cls_scores is None:
            detached_cls_scores = [None] * num_imgs
        else:
            detached_cls_scores = cls_scores.detach()

        for i in range(num_imgs):
            assign_result = self.assigner.assign(scaled_mask_preds[i].detach(),
                                                 detached_cls_scores[i],
                                                 gt_masks[i], gt_labels[i],
                                                 img_metas[i])
            sampling_result = self.sampler.sample(assign_result,
                                                  scaled_mask_preds[i],
                                                  gt_masks[i])
            sampling_results.append(sampling_result)
        
        num_batch = scaled_mask_preds.shape[0]
        
        #print("num_batch-scaled_mask_preds.shape[0]", num_batch)
        mask_targets = self.get_targets(
            sampling_results,
            gt_masks,
            self.train_cfg,
            True,
            num_batch,
            gt_scores,
            gt_ious,
            flag,
            gt_sem_seg=gt_sem_seg,
            gt_sem_cls=gt_sem_cls)
        
        
#------------------------------------#        
        #labels, label_weights, mask_targets, mask_weights, seg_targets = mask_targets
        #print("labels.shape", labels.shape)
        #print("scaled_mask_preds.shape:", scaled_mask_preds.shape)
        #!对数据进行判断，如果是无标签数据则使用rce_loss

        losses = self.loss(scaled_mask_preds, cls_scores, scaled_seg_preds,
                           proposal_feats, flag, *mask_targets)

        if self.cat_stuff_mask and self.training:
            mask_preds = torch.cat(
                [mask_preds, seg_preds[:, self.num_thing_classes:]], dim=1)
            stuff_kernels = self.conv_seg.weight[self.
                                                 num_thing_classes:].clone()
            stuff_kernels = stuff_kernels[None].expand(num_imgs,
                                                       *stuff_kernels.size())
            proposal_feats = torch.cat([proposal_feats, stuff_kernels], dim=1)

        return losses, proposal_feats, x_feats, mask_preds, cls_scores

    
    @force_fp32(apply_to=('cls_score', 'mask_pred', 'seg_preds'))
    def loss(self,
             mask_pred,
             cls_scores,
             seg_preds,
             proposal_feats,
             flag,
             labels,
             label_weights,
             mask_targets,
             mask_weights,
             seg_targets,
             reduction_override=None,
             **kwargs):
        losses = dict()
        bg_class_ind = self.num_classes
        # note in spare rcnn num_gt == num_pos
        pos_inds = (labels >= 0) & (labels < bg_class_ind)

#------------------------------------------#        
        #print("pos_inds.shape:", pos_inds.shape)
        #print("labels.shape", labels.shape)
        loss_mask = self.loss_mask
        '''
        if flag==1:
            loss_mask = self.loss_mask
        else:
            loss_mask = self.loss_mask_1
        '''
        num_preds = mask_pred.shape[0] * mask_pred.shape[1]
        
        if cls_scores is not None:
            num_pos = pos_inds.sum().float()
            avg_factor = reduce_mean(num_pos)
            assert mask_pred.shape[0] == cls_scores.shape[0]
            assert mask_pred.shape[1] == cls_scores.shape[1]
            losses['loss_rpn_cls'] = self.loss_cls(
                cls_scores.view(num_preds, -1),
                labels,
                label_weights,
                avg_factor=avg_factor,
                reduction_override=reduction_override)
            losses['rpn_pos_acc'] = accuracy(
                cls_scores.view(num_preds, -1)[pos_inds], labels[pos_inds])

        bool_pos_inds = pos_inds.type(torch.bool)
        # 0~self.num_classes-1 are FG, self.num_classes is BG
        # do not perform bounding box regression for BG anymore.
        H, W = mask_pred.shape[-2:]
        if pos_inds.any():
#-------------------------------------------------#            
            
            #print("mask_pred.shape", mask_pred.shape)  #torch.Size([3, 100, 256, 336])
            #print("bool_pos_inds.shape", bool_pos_inds.shape) #torch.Size([200])
            #print("+"*100)
            #print("loss_mask:", loss_mask)
            
            pos_mask_pred = mask_pred.reshape(num_preds, H, W)[bool_pos_inds]
            pos_mask_targets = mask_targets[bool_pos_inds]
            pos_mask_weight = mask_weights[bool_pos_inds]
            h, w = mask_targets.shape[-2:]
            num_pos = pos_mask_weight.shape[0]
            pos_mask_weight_1 = pos_mask_weight.unsqueeze(1).unsqueeze(1).expand([num_pos, h, w])
            '''
            if flag==1:
                print("h_loss"+"-"*100)
                mm = pos_mask_weight.detach().cpu().numpy()
                #mt = pos_mask_targets.detach().cpu().numpy()
                print("pos_mask_weight",mm, pos_mask_weight.shape)
            
            if flag==0:
                print("head"+"-"*100)
                #print("pos_mask_weight_1",pos_mask_weight_1, pos_mask_weight_1.shape)
                #rint("H, W", H, W)
                #print("pos_mask_targets", mt, pos_mask_targets.shape)
                import numpy as np
                #np.set_printoptions(threshold=np.inf)
                print("rpn_loss"+"-"*100)
                mm = pos_mask_weight_1.detach().cpu().numpy()
                #mt = mask_targets.detach().cpu().numpy()
                print("pos_mask_weight_1", mm, pos_mask_weight_1.shape)
            '''
            
            losses['loss_rpn_mask'] = loss_mask(pos_mask_pred,
                                                     pos_mask_targets,
                                               pos_mask_weight_1)
            losses['loss_rpn_dice'] = self.loss_dice(pos_mask_pred,
                                                     pos_mask_targets,
                                                    pos_mask_weight)
#---------------------------------#
            #print("-"*100)
            #print(loss_mask, losses['loss_rpn_mask'])
    
    
            if self.loss_rank is not None:
                batch_size = mask_pred.size(0)
                rank_target = mask_targets.new_full((batch_size, H, W),
                                                    self.ignore_label,
                                                    dtype=torch.long)
                rank_inds = pos_inds.view(batch_size,
                                          -1).nonzero(as_tuple=False)
                batch_mask_targets = mask_targets.view(batch_size, -1, H,
                                                       W).bool()
                for i in range(batch_size):
                    curr_inds = (rank_inds[:, 0] == i)
                    curr_rank = rank_inds[:, 1][curr_inds]
                    for j in curr_rank:
                        rank_target[i][batch_mask_targets[i][j]] = j
                losses['loss_rpn_rank'] = self.loss_rank(
                    mask_pred, rank_target, ignore_index=self.ignore_label)

        else:
            losses['loss_rpn_mask'] = mask_pred.sum() * 0
            losses['loss_rpn_dice'] = mask_pred.sum() * 0
            if self.loss_rank is not None:
                losses['loss_rank'] = mask_pred.sum() * 0

        if seg_preds is not None:
#----------------------------#
            #print("seg_targets:", seg_targets)
            if self.loss_seg.use_sigmoid:
                cls_channel = seg_preds.shape[1]
                flatten_seg = seg_preds.view(
                    -1, cls_channel,
                    H * W).permute(0, 2, 1).reshape(-1, cls_channel)

#---------------------------------#
                #print("flatten_seg.shape:", flatten_seg.shape) #torch.Size([60800, 80])
                flatten_seg_target = seg_targets.view(-1)
                num_dense_pos = (flatten_seg_target >= 0) & (
                    flatten_seg_target < bg_class_ind)
                num_dense_pos = num_dense_pos.sum().float().clamp(min=1.0)
#!无标签图片无mask时不应该计算rpn_seg这个loss，因为flatten_seg_target是全零
#---------------------------------------------------------------------#
                if pos_inds.any():
                    losses['loss_rpn_seg'] = self.loss_seg(
                        flatten_seg,
                        flatten_seg_target,
                        avg_factor=num_dense_pos)
                else:
                    losses['loss_rpn_seg'] = seg_preds.sum() * 0
            else:
                cls_channel = seg_preds.shape[1]
                flatten_seg = seg_preds.view(-1, cls_channel, H * W).permute(
                    0, 2, 1).reshape(-1, cls_channel)
                flatten_seg_target = seg_targets.view(-1)

                losses['loss_rpn_seg'] = self.loss_seg(flatten_seg,
                                                       flatten_seg_target)

        return losses

    def _get_target_single(self, pos_inds, neg_inds, pos_mask, neg_mask,
                           pos_gt_mask, pos_gt_labels, pos_score, pos_iou, flag, gt_sem_seg, gt_sem_cls,
                           cfg):
        num_pos = pos_mask.size(0)
        num_neg = neg_mask.size(0)
        num_samples = num_pos + num_neg
        H, W = pos_mask.shape[-2:]
        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        labels = pos_mask.new_full((num_samples, ),
                                   self.num_classes,
                                   dtype=torch.long)
        label_weights = pos_mask.new_zeros(num_samples)
        mask_targets = pos_mask.new_zeros(num_samples, H, W)
        #mask_weights = pos_mask.new_zeros(num_samples, H, W)
        mask_weights = pos_mask.new_zeros(num_samples)
        seg_targets = pos_mask.new_full((H, W),
                                        self.num_classes,
                                        dtype=torch.long)

#--------------------------------#
        #print("gt_sem_cls:", gt_sem_cls, gt_sem_seg) #None
        #print("seg_targets:", seg_targets)


        if gt_sem_cls is not None and gt_sem_seg is not None:
            gt_sem_seg = gt_sem_seg.bool()
            for sem_mask, sem_cls in zip(gt_sem_seg, gt_sem_cls):
                seg_targets[sem_mask] = sem_cls.long()

        if num_pos > 0:
            labels[pos_inds] = pos_gt_labels
            
#修改权重，分类采用score，mask采用iou--------------------------------#            
            #pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight #cfg 1
            if flag == 1:
                pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
                label_weights[pos_inds] = pos_weight 
            else:
                #label_weights[pos_inds] = pos_score * pos_score
#修改分类的权重，>=0.5之后就不再使用动态阈值                
                pos_score = pos_score.clamp(0, 0.5)
                   
        
                #label_weights[pos_inds] = pos_score * pos_score * pos_score * pos_score * 12                
                label_weights[pos_inds] = 1
                
                #print("检查是否需要梯度"+"="*50)
                #print("label_weights", label_weights, label_weights.requires_grad) false
            #print("pos_inds", pos_inds)
            #print("pos_gt_mask", pos_gt_mask)
            #print("mask_targets", mask_targets)
            #print("+"*100)
            #print("pos_gt_mask.dtype", pos_gt_mask.dtype)    #torch.float32
            #print("mask_targets.dtype", mask_targets.dtype)  #torch.float16
            
            mask_targets[pos_inds, ...] = pos_gt_mask
            if flag == 1:
                #mask_weights[pos_inds, ...] = 1
                mask_weights[pos_inds] = 1
            else:
                assert num_pos == pos_iou.shape[0]
                #pos_iou = pos_iou.unsqueeze(1).unsqueeze(1).expand([num_pos, H, W])
                #mask_weights[pos_inds, ...] = pos_iou
              
                #mask_weights[pos_inds] = pos_iou * pos_iou
                #mask_weights[pos_inds] = pos_iou * pos_iou * pos_iou * pos_iou * 2.5
                mask_weights[pos_inds] = 1
                
                '''
                import numpy as np
                #np.set_printoptions(threshold=np.inf)
                print("rpn_get"+"-"*100)
                mm = mask_weights.detach().cpu().numpy()
                mt = mask_targets.detach().cpu().numpy()
                print("mask_weights", mm, mask_weights.shape)
                #print("mask_targets", mt)
                '''
            for i in range(num_pos):
                seg_targets[pos_gt_mask[i].bool()] = pos_gt_labels[i]

#这边是否也需要改成对应的score?????????????????????                
        if num_neg > 0:
            if flag == 0:
                #label_weights[neg_inds] = 0.35 *0.35
                label_weights[neg_inds] = 1.0
            else:
                label_weights[neg_inds] = 1.0
            
#------------------------------------#        
        #print("labels:", labels.shape)
        
        return labels, label_weights, mask_targets, mask_weights, seg_targets

    def get_targets(self,
                    sampling_results,
                    gt_mask,
                    rpn_train_cfg,
                    concat=True,
                    num=1,
                    gt_scores=None,
                    gt_ious=None,
                    flag=1,
                    gt_sem_seg=None,
                    gt_sem_cls=None):
        pos_inds_list = [res.pos_inds for res in sampling_results]
        neg_inds_list = [res.neg_inds for res in sampling_results]
        pos_mask_list = [res.pos_masks for res in sampling_results]
        neg_mask_list = [res.neg_masks for res in sampling_results]
        pos_gt_mask_list = [res.pos_gt_masks for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        pos_assigned_gt_inds_list = [res.pos_assigned_gt_inds for res in sampling_results]

        if flag == 0:
            pos_box_mask_list = [res.pos_box_mask for res in sampling_results]
        
#此处取pos iou和score，是否应该去梯度？？？？？？？？？   
        
        pos_score_list = []
        pos_iou_list = []
        num_image = len(pos_assigned_gt_inds_list)
        
        assert num_image == num
        
        if flag == 0:
            for i in range(num_image):
                if pos_assigned_gt_inds_list[i].numel() == 0:
                    pos_score = None
                    pos_iou = None
                else:
                    pos_score = gt_scores[i][pos_assigned_gt_inds_list[i]]
                    pos_iou = gt_ious[i][pos_assigned_gt_inds_list[i]]
                pos_score_list.append(pos_score)
                pos_iou_list.append(pos_iou)
        #print("len(pos_inds_list)", len(pos_inds_list))
        #print("len(pos_gt_mask_list)", len(pos_gt_mask_list))
        flag_list = [flag for i in range(num_image)]
        #print("+"*100)
        #print("pos_gt_mask_list[0].dtype:", pos_gt_mask_list[0].dtype) #torch.float32
        #print("pos_mask_list[0].dtype:", pos_mask_list[0].dtype) #torch.float16
        
        if gt_sem_seg is None:
            #gt_sem_seg = [None] * 2
            gt_sem_seg = [None] * num
            gt_sem_cls = [None] * num
            #gt_sem_cls = [None] * 2
        if gt_scores is None:
            pos_score_list = [None] * num
            pos_iou_list = [None] * num
        
    
        results = multi_apply(
            self._get_target_single,
            pos_inds_list,
            neg_inds_list,
            pos_mask_list,
            neg_mask_list,
            pos_gt_mask_list,
            pos_gt_labels_list,
            pos_score_list,
            pos_iou_list,
            flag_list,
            gt_sem_seg,
            gt_sem_cls,
            cfg=rpn_train_cfg)
        (labels, label_weights, mask_targets, mask_weights,
         seg_targets) = results
        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            mask_targets = torch.cat(mask_targets, 0)
            mask_weights = torch.cat(mask_weights, 0)
            seg_targets = torch.stack(seg_targets, 0)
        if flag == 0:
            return labels, label_weights, mask_targets, mask_weights, seg_targets, pos_box_mask_list, pos_mask_list
        else: 
            return labels, label_weights, mask_targets, mask_weights, seg_targets

    def simple_test_rpn(self, img, img_metas):
        """Forward function in testing stage."""
        return self._decode_init_proposals(img, img_metas)

    def forward_dummy(self, img, img_metas):
        """Dummy forward function.

        Used in flops calculation.
        """
        return self._decode_init_proposals(img, img_metas)
    
    @force_fp32(apply_to=('img', 'gt_masks')) 
    def forward_train_unsup(self,
                      img,
                      img_metas,
                      gt_masks,
                      gt_box_masks,
                      gt_labels,
                      gt_scores=None,
                      gt_ious=None,
                      gt_sem_seg=None,
                      gt_sem_cls=None,
                      img_color_similarity=None):
        """Forward function in training stage."""
        num_imgs = len(img_metas)
        results = self._decode_init_proposals(img, img_metas)
        (proposal_feats, x_feats, mask_preds, cls_scores, seg_preds) = results

#----------------------------------------#        
        #print("mask_preds.dtype", mask_preds.dtype)  #torch.float16
        #print("seg_preds.shape", seg_preds.dtype, seg_preds.shape) #torch.Size([1, 80, 152, 100])
        #print("+"*100)
        #print("gt_masks[0].dtype", gt_masks[0].dtype) #torch.float16
        
        if self.feat_downsample_stride > 1:
            scaled_mask_preds = F.interpolate(
                mask_preds,
                scale_factor=self.feat_downsample_stride,
                mode='bilinear',
                align_corners=False)
            if seg_preds is not None:
                scaled_seg_preds = F.interpolate(
                    seg_preds,
                    scale_factor=self.feat_downsample_stride,
                    mode='bilinear',
                    align_corners=False)
        else:
            scaled_mask_preds = mask_preds
            scaled_seg_preds = seg_preds

        if self.hard_target:
            gt_masks = [x.bool().float() for x in gt_masks]
        else:
            gt_masks = gt_masks
            
        
#------------------------------#      
        """
        print("scaled_mask_preds.shape:", scaled_mask_preds.shape)   #torch.Size([1, 100, 240, 320])
        print("gt_masks.shape:", len(gt_masks), gt_masks[0].shape)   #1 torch.Size([5, 240, 320])
        print("gt_labels.shape", len(gt_labels), gt_labels[0].shape)
        print("gt_labels:", gt_labels)  
        """
        #print("scaled_mask_preds.shape:", scaled_mask_preds.shape)  
        #print("gt_labels.shape", len(gt_labels), gt_labels[0].shape)  #3 torch.Size([3]) 同mask_pred
        #print("scaled_mask_preds.dtype:", scaled_mask_preds.dtype) #torch.float16
        #print("gt_masks.dtype:", gt_masks[0].dtype)  #torch.float32
        #scaled_mask_preds = scaled_mask_preds.float()
        #print("scaled_mask_preds.dtype:", scaled_mask_preds.dtype)
        #print("gt_sem_seg.shape:", len(gt_sem_seg), gt_sem_seg[0].shape) #None
        #print("gt_sem_cls.shape:", len(gt_sem_cls), gt_sem_cls[0].shape) #None
        flag = 1
        if img_metas[0]["tag"] == "sup":
            flag = 1
        else:
            flag = 0
        
    
        sampling_results = []
        if cls_scores is None:
            detached_cls_scores = [None] * num_imgs
        else:
            detached_cls_scores = cls_scores.detach()

        for i in range(num_imgs):
            assign_result = self.assigner.assign(scaled_mask_preds[i].detach(),
                                                 detached_cls_scores[i],
                                                 gt_masks[i], gt_labels[i],
                                                 img_metas[i])
            sampling_result = self.sampler.sample(assign_result,
                                                  scaled_mask_preds[i],
                                                  gt_masks[i], gt_box_masks[i])
            sampling_results.append(sampling_result)
        
        num_batch = scaled_mask_preds.shape[0]

        #level_set_feat = self.feat_conv(img[0])
        
        #print("num_batch-scaled_mask_preds.shape[0]", num_batch)
        mask_targets = self.get_targets(
            sampling_results,
            gt_masks,
            self.train_cfg,
            True,
            num_batch,
            gt_scores,
            gt_ious,
            flag,
            gt_sem_seg=gt_sem_seg,
            gt_sem_cls=gt_sem_cls)
        
        
#------------------------------------#        
        #labels, label_weights, mask_targets, mask_weights, seg_targets = mask_targets
        #print("labels.shape", labels.shape)
        #print("scaled_mask_preds.shape:", scaled_mask_preds.shape)
        #!对数据进行判断，如果是无标签数据则使用rce_loss

        losses = self.loss(scaled_mask_preds, cls_scores, scaled_seg_preds,
                           proposal_feats, flag, *mask_targets)
        
        #losses = self.loss_ls(losses, level_set_feat, mask_targets[-2], mask_targets[-1])

#计算pairwise loss-------------------------
        losses = self.loss_bi(losses, scaled_mask_preds, img_color_similarity, mask_targets[-2], mask_targets[-1])

        if self.cat_stuff_mask and self.training:
            mask_preds = torch.cat(
                [mask_preds, seg_preds[:, self.num_thing_classes:]], dim=1)
            stuff_kernels = self.conv_seg.weight[self.
                                                 num_thing_classes:].clone()
            stuff_kernels = stuff_kernels[None].expand(num_imgs,
                                                       *stuff_kernels.size())
            proposal_feats = torch.cat([proposal_feats, stuff_kernels], dim=1)

        return losses, proposal_feats, x_feats, mask_preds, cls_scores

    @force_fp32(apply_to=('pos_mask_list', 'pos_box_mask_list'))
    def loss_bi(self,
             loss,
             mask_preds,
             img_color_similarity,
             pos_box_mask_list, 
             pos_mask_list
            ):

#准备计算pairwise loss的color_similarity和box_mask，要求把第一维度的batch_size，扩充成伪标签实例的总和
        all_instance_color_similarity = []
        all_mask_pred = []
        all_box_mask = []
        flag_bt = 0 # flag判断batch图片内有无伪标签，没有的话直接就不算这个loss了，把loss置0并返回
        for i in range(len(pos_mask_list)):
            if pos_mask_list[i].size(0) != 0:
                flag_bt = 1
                perimg_color_similarity = torch.cat([
                    img_color_similarity[i][None] for _ in range(pos_mask_list[i].size(0))
                ], dim=0)
                all_instance_color_similarity.append(perimg_color_similarity)
                all_mask_pred.append(pos_mask_list[i])
                all_box_mask.append(pos_box_mask_list[i])
        if flag_bt == 1:
            all_instance_color_similarity = torch.cat(all_instance_color_similarity, dim = 0)
            all_mask_pred = torch.cat(all_mask_pred, dim = 0)
            all_box_mask = torch.cat(all_box_mask, dim = 0)
            loss_bi = self.loss_pairwise(all_instance_color_similarity, all_mask_pred, all_box_mask, flag_bt)
            loss['loss_rpn_pairwise'] = loss_bi
        else:
            loss['loss_rpn_pairwise'] = mask_preds.sum() * 0

        return loss


    @force_fp32(apply_to=('pos_mask_list', 'level_set_feat'))
    def loss_ls(self,
             loss,
             level_set_feat,
             pos_box_mask_list, 
             pos_mask_list
            ):
        loss_levelset = []
        sum_pos = 0
        for i in range(level_set_feat.shape[0]):
            mask_pred = pos_mask_list[i].unsqueeze(dim=1)
            box_mask_target = pos_box_mask_list[i].unsqueeze(dim=1).to(dtype=mask_pred.dtype)
            feat = level_set_feat[i].unsqueeze(dim=0).expand(mask_pred.shape[0], level_set_feat[i].shape[0], level_set_feat[i].shape[1], level_set_feat[i].shape[2])
            if mask_pred.shape[0] == 0:
                loss_img_lst = feat.sum() * 0
                loss_img_lst = loss_img_lst.unsqueeze(0)
            else:
                sum_pos += mask_pred.shape[0]
                mask_pred = torch.sigmoid(mask_pred)
                back_scores = 1.0 - mask_pred
                mask_scores_concat = torch.cat((mask_pred, back_scores), dim=1)
                mask_scores_phi = mask_scores_concat * box_mask_target
                img_target_wbox = feat * box_mask_target
                pixel_num = box_mask_target.sum((1, 2, 3))
                pixel_num = torch.clamp(pixel_num, min=1)
                loss_img_lst = self.loss_levelset(mask_scores_phi, img_target_wbox, pixel_num) * 0.25

            loss_levelset.append(loss_img_lst)
        #sum_pos = torch.clip(torch.tensor([sum_pos], device = level_set_feat[0].device), min = 1)
        if sum_pos == 0:
            sum_pos = 1
        loss_levelset = torch.cat(loss_levelset).sum() / sum_pos
        loss['loss_rpn_levelset'] = loss_levelset
        return loss