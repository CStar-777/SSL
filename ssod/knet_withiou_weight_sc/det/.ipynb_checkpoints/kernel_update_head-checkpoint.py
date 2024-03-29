import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (ConvModule, bias_init_with_prob, build_activation_layer,
                      build_norm_layer)
from mmcv.cnn.bricks.transformer import (FFN, MultiheadAttention,
                                         build_transformer_layer)
from mmcv.runner import force_fp32, BaseModule

from mmdet.core import multi_apply, mask_matrix_nms
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.dense_heads.atss_head import reduce_mean
from mmdet.models.losses import accuracy
from mmdet.utils import get_root_logger
from .utils import compute_mask_iou


@HEADS.register_module()
class KernelUpdateHead(BaseModule):

    def __init__(self,
                 num_classes=80,
                 num_ffn_fcs=2,
                 num_heads=8,
                 num_cls_fcs=1,
                 num_mask_fcs=3,
                 feedforward_channels=2048,
                 in_channels=256,
                 out_channels=256,
                 dropout=0.0,
                 mask_thr=0.5,
                 act_cfg=dict(type='ReLU', inplace=True),
                 ffn_act_cfg=dict(type='ReLU', inplace=True),
                 conv_kernel_size=3,
                 feat_transform_cfg=None,
                 hard_mask_thr=0.5,
                 kernel_init=False,
                 with_ffn=True,
                 mask_out_stride=4,
                 relative_coors=False,
                 relative_coors_off=False,
                 feat_gather_stride=1,
                 mask_transform_stride=1,
                 mask_upsample_stride=1,
                 num_thing_classes=80,
                 num_stuff_classes=53,
                 mask_assign_stride=4,
                 ignore_label=255,
                 thing_label_in_seg=0,
                 kernel_updator_cfg=dict(
                     type='DynamicConv',
                     in_channels=256,
                     feat_channels=64,
                     out_channels=256,
                     input_feat_shape=1,
                     act_cfg=dict(type='ReLU', inplace=True),
                     norm_cfg=dict(type='LN')),
                 loss_rank=None,
                 loss_mask=dict(
                     type='CrossEntropyLoss', use_mask=True, loss_weight=1.0),
                 loss_mask_1=None,                
                 loss_dice=dict(type='DiceLoss', loss_weight=3.0),
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=2.0),
                 loss_iou=dict(
                     type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
                 init_cfg=None,
                 ):
        super(KernelUpdateHead, self).__init__(init_cfg)
        self.num_classes = num_classes
        self.loss_cls = build_loss(loss_cls)
        self.loss_mask = build_loss(loss_mask)
        #self.loss_mask_1 = build_loss(loss_mask_1)
        self.loss_dice = build_loss(loss_dice)
        if loss_rank is not None:
            self.loss_rank = build_loss(loss_rank)
        else:
            self.loss_rank = loss_rank
        self.loss_iou = build_loss(loss_iou)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mask_thr = mask_thr
        self.fp16_enabled = False
        self.dropout = dropout

        self.num_heads = num_heads
        self.hard_mask_thr = hard_mask_thr
        self.kernel_init = kernel_init
        self.with_ffn = with_ffn
        self.mask_out_stride = mask_out_stride
        self.relative_coors = relative_coors
        self.relative_coors_off = relative_coors_off
        self.conv_kernel_size = conv_kernel_size
        self.feat_gather_stride = feat_gather_stride
        self.mask_transform_stride = mask_transform_stride
        self.mask_upsample_stride = mask_upsample_stride

        self.num_thing_classes = num_thing_classes
        self.num_stuff_classes = num_stuff_classes
        self.mask_assign_stride = mask_assign_stride
        self.ignore_label = ignore_label
        self.thing_label_in_seg = thing_label_in_seg

        self.attention = MultiheadAttention(in_channels * conv_kernel_size**2,
                                            num_heads, dropout)
        self.attention_norm = build_norm_layer(
            dict(type='LN'), in_channels * conv_kernel_size**2)[1]

        self.kernel_update_conv = build_transformer_layer(kernel_updator_cfg)

        if feat_transform_cfg is not None:
            kernel_size = feat_transform_cfg.pop('kernel_size', 1)
            self.feat_transform = ConvModule(
                in_channels,
                in_channels,
                kernel_size,
                stride=feat_gather_stride,
                padding=int(feat_gather_stride // 2),
                **feat_transform_cfg)
        else:
            self.feat_transform = None

        if self.with_ffn:
            self.ffn = FFN(
                in_channels,
                feedforward_channels,
                num_ffn_fcs,
                act_cfg=ffn_act_cfg,
                dropout=dropout)
            self.ffn_norm = build_norm_layer(dict(type='LN'), in_channels)[1]

        self.cls_fcs = nn.ModuleList()
        for _ in range(num_cls_fcs):
            self.cls_fcs.append(
                nn.Linear(in_channels, in_channels, bias=False))
            self.cls_fcs.append(
                build_norm_layer(dict(type='LN'), in_channels)[1])
            self.cls_fcs.append(build_activation_layer(act_cfg))
          
#!增加一个预测iou的模块
        self.iou_fcs = nn.ModuleList()
        for _ in range(1):
            self.iou_fcs.append(
                nn.Linear(in_channels, in_channels, bias=False))
            self.iou_fcs.append(
                build_norm_layer(dict(type='LN'), in_channels)[1])
            self.iou_fcs.append(build_activation_layer(act_cfg))
      
        if self.loss_cls.use_sigmoid:
            self.fc_cls = nn.Linear(in_channels, self.num_classes)
        else:
            self.fc_cls = nn.Linear(in_channels, self.num_classes + 1)

        self.mask_fcs = nn.ModuleList()
        for _ in range(num_mask_fcs):
            self.mask_fcs.append(
                nn.Linear(in_channels, in_channels, bias=False))
            self.mask_fcs.append(
                build_norm_layer(dict(type='LN'), in_channels)[1])
            self.mask_fcs.append(build_activation_layer(act_cfg))

        self.fc_mask = nn.Linear(in_channels, out_channels)

#!增加一个分支，来预测iou的分数-----------------------#
        self.fc_objectness = nn.Linear(in_channels, 1)
        
        if self.init_cfg is None:
            self._init_weights()
        

    
    def _init_weights(self):
        #print("使用了自定义权重初始化")
       #! Use xavier initialization for all weight parameter and set
       # classification head bias as a specific value when use focal loss.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                # adopt the default initialization for
                # the weight and bias of the layer norm
                pass
        
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            nn.init.constant_(self.fc_cls.bias, bias_init)
        if self.kernel_init:
            logger = get_root_logger()
            logger.info(
                'mask kernel in mask head is normal initialized by std 0.01')
            nn.init.normal_(self.fc_mask.weight, mean=0, std=0.01)


    @force_fp32(apply_to=('x_feat', 'proposal_feat'))
    def cal(self,
            x_feat,
            proposal_feat):
        obj_feat = self.kernel_update_conv(x_feat, proposal_feat)
        
        #print("obj_feat.dtype", obj_feat.dtype) #torch.float32
        return obj_feat
   

    @force_fp32(apply_to=('x', 'proposal_feat', 'mask_preds'))
    def forward(self,
                x,
                proposal_feat,
                mask_preds,
                prev_cls_score=None,
                mask_shape=None,
                img_metas=None):       

        #print("+"*100)
        #print("mask_preds", mask_preds) #正常
        #print("x", x)
        #print("proposal_feat", proposal_feat)
        
        N, num_proposals = proposal_feat.shape[:2]
        if self.feat_transform is not None:
            x = self.feat_transform(x)
        C, H, W = x.shape[-3:]

        mask_h, mask_w = mask_preds.shape[-2:]
        if mask_h != H or mask_w != W:
            gather_mask = F.interpolate(
                mask_preds, (H, W), align_corners=False, mode='bilinear')
        else:
            gather_mask = mask_preds

        sigmoid_masks = gather_mask.sigmoid()
        nonzero_inds = sigmoid_masks > self.hard_mask_thr
        sigmoid_masks = nonzero_inds.float()

        # einsum is faster than bmm by 30%
        x_feat = torch.einsum('bnhw,bchw->bnc', sigmoid_masks, x)
        
        #print("|"*100)
        #print("x_feat", x_feat) #正常，但是数值较大

        # obj_feat in shape [B, N, C, K, K] -> [B, N, C, K*K] -> [B, N, K*K, C]
        proposal_feat = proposal_feat.reshape(N, num_proposals,
                                              self.in_channels,
                                              -1).permute(0, 1, 3, 2)
#!修改，将这边的计算改成单精度的    
#---------------------------------------#
        obj_feat = self.kernel_update_conv(x_feat, proposal_feat)
        #obj_feat = self.cal(x_feat, proposal_feat)
        
        #print("|"*100)
        #print("obj_feat.dtype", obj_feat.dtype) #torch.float32
        #print("obj_feat", obj_feat) #nan

        # [B, N, K*K, C] -> [B, N, K*K*C] -> [N, B, K*K*C]
        obj_feat = obj_feat.reshape(N, num_proposals, -1).permute(1, 0, 2)
        obj_feat = self.attention_norm(self.attention(obj_feat))
        # [N, B, K*K*C] -> [B, N, K*K*C]
        obj_feat = obj_feat.permute(1, 0, 2)

        # obj_feat in shape [B, N, K*K*C] -> [B, N, K*K, C]
        obj_feat = obj_feat.reshape(N, num_proposals, -1, self.in_channels)

        #print("-"*100)
        #print("obj_feat", obj_feat)
        
        # FFN
        if self.with_ffn:
            obj_feat = self.ffn_norm(self.ffn(obj_feat))

        #print("+"*100)
        #print("obj_feat", obj_feat)
           
        cls_feat = obj_feat.sum(-2) #b, n, c
        mask_feat = obj_feat
        iou_feat =  obj_feat.sum(-2) #b, n, c
        
        for iou_layer in self.iou_fcs:
            iou_feat = iou_layer(iou_feat)

        for cls_layer in self.cls_fcs:
            cls_feat = cls_layer(cls_feat)
        for reg_layer in self.mask_fcs:
            mask_feat = reg_layer(mask_feat)
            

        cls_score = self.fc_cls(cls_feat).view(N, num_proposals, -1)
#!利用kernel的特征进行预测iou的分值--------------
#-------------------------------------------------#
        #print("+"*100)
        #print("cls_score.shape", cls_score.shape) #torch.Size([2, 100, 80])
        iou_scores = self.fc_objectness(iou_feat)
        #print("iou_scores", iou_scores) 
        #print("iou_scores.shape", iou_scores.shape) #torch.Size([2, 100, 1])
        
        # [B, N, K*K, C] -> [B, N, C, K*K]
        mask_feat = self.fc_mask(mask_feat).permute(0, 1, 3, 2)

        if (self.mask_transform_stride == 2 and self.feat_gather_stride == 1):
            mask_x = F.interpolate(
                x, scale_factor=0.5, mode='bilinear', align_corners=False)
            H, W = mask_x.shape[-2:]
        else:
            mask_x = x
        # group conv is 5x faster than unfold and uses about 1/5 memory
        # Group conv vs. unfold vs. concat batch, 2.9ms :13.5ms :3.8ms
        # Group conv vs. unfold vs. concat batch, 278 : 1420 : 369
        # fold_x = F.unfold(
        #     mask_x,
        #     self.conv_kernel_size,
        #     padding=int(self.conv_kernel_size // 2))
        # mask_feat = mask_feat.reshape(N, num_proposals, -1)
        # new_mask_preds = torch.einsum('bnc,bcl->bnl', mask_feat, fold_x)
        # [B, N, C, K*K] -> [B*N, C, K, K]
        mask_feat = mask_feat.reshape(N, num_proposals, C,
                                      self.conv_kernel_size,
                                      self.conv_kernel_size)
        # [B, C, H, W] -> [1, B*C, H, W]
        new_mask_preds = []
        for i in range(N):
            new_mask_preds.append(
                F.conv2d(
                    mask_x[i:i + 1],
                    mask_feat[i],
                    padding=int(self.conv_kernel_size // 2)))

        new_mask_preds = torch.cat(new_mask_preds, dim=0)
        new_mask_preds = new_mask_preds.reshape(N, num_proposals, H, W)
        if self.mask_transform_stride == 2:
            new_mask_preds = F.interpolate(
                new_mask_preds,
                scale_factor=2,
                mode='bilinear',
                align_corners=False)

        if mask_shape is not None and mask_shape[0] != H:
            new_mask_preds = F.interpolate(
                new_mask_preds,
                mask_shape,
                align_corners=False,
                mode='bilinear')
#-----------------------#
        #print("+"*100)
        #print("cls_score", cls_score) #nan torch.float16
        #print("new_mask_preds", new_mask_preds)     #nan torch.float16
            
            
        return cls_score, iou_scores, new_mask_preds, obj_feat.permute(0, 1, 3, 2).reshape(
            N, num_proposals, self.in_channels, self.conv_kernel_size,
            self.conv_kernel_size)

    @force_fp32(apply_to=('cls_score', 'mask_pred'))
    def loss(self,
             object_feats,
             cls_score,
             iou_scores,
             mask_pred,
             flag,
             is_scale,
             labels,
             label_weights,
             mask_targets,
             mask_weights,
             imgs_whwh=None,
             reduction_override=None,
             **kwargs):
        if is_scale == None:
            str_loss_cls = "loss_cls"
            str_pos_acc = "pos_acc"
            str_loss_mask = "loss_mask"
            str_loss_dice = "loss_dice"
            str_loss_iou = "loss_iou"
        else:
            str_loss_cls = "loss_cls" + is_scale
            str_pos_acc = "pos_acc" + is_scale
            str_loss_mask = "loss_mask" + is_scale
            str_loss_dice = "loss_dice" + is_scale
            str_loss_iou = "loss_iou" + is_scale

        loss_mask = self.loss_mask
        '''
        if flag==1:
            loss_mask = self.loss_mask
        else:
            loss_mask = self.loss_mask_1
        '''
#----------------------------#
        #print("+"*100)
        #print("loss_mask:", loss_mask)
        
        losses = dict()
        bg_class_ind = self.num_classes
        # note in spare rcnn num_gt == num_pos
        pos_inds = (labels >= 0) & (labels < bg_class_ind)
        num_pos = pos_inds.sum().float()
        avg_factor = reduce_mean(num_pos).clamp_(min=1.0)

        num_preds = mask_pred.shape[0] * mask_pred.shape[1]
        assert mask_pred.shape[0] == cls_score.shape[0]
        assert mask_pred.shape[1] == cls_score.shape[1]

        if cls_score is not None:
            if cls_score.numel() > 0:
                if flag==0:
                    #print("head_loss"+"|"*50)

                    #cl = cls_score.view(num_preds, -1).detach().cpu().numpy()
                    la = labels.detach().cpu().numpy()
                    lw = label_weights.detach().cpu().numpy()
                    '''
                    print("head_get"+"="*100)
                    import numpy as np
                    np.set_printoptions(threshold=np.inf)
                    #print("cls_score.view(num_preds, -1)", cl)
                    print("labels", la)
                    print("label_weights", lw)
                    '''
                losses[str_loss_cls] = self.loss_cls(
                    cls_score.view(num_preds, -1),
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                losses[str_pos_acc] = accuracy(
                    cls_score.view(num_preds, -1)[pos_inds], labels[pos_inds])
        
        if mask_pred is not None:
            bool_pos_inds = pos_inds.type(torch.bool)
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            # do not perform bounding box regression for BG anymore.
            H, W = mask_pred.shape[-2:]
            if pos_inds.any():
                pos_mask_pred = mask_pred.reshape(num_preds, H,
                                                  W)[bool_pos_inds]
                pos_mask_targets = mask_targets[bool_pos_inds]
                pos_mask_weight = mask_weights[bool_pos_inds]
                h, w = mask_targets.shape[-2:]
                num_pos = pos_mask_weight.shape[0]
                pos_mask_weight_1 = pos_mask_weight.unsqueeze(1).unsqueeze(1).expand([num_pos, h, w])
                
                #print("head_loss"+"|"*50)
                #print("pos_mask_weight", pos_mask_weight, pos_mask_weight.shape)
                #print("pos_mask_targets", pos_mask_targets, pos_mask_targets.shape)
                
                losses[str_loss_mask] = loss_mask(pos_mask_pred,
                                                     pos_mask_targets,
                                               pos_mask_weight_1)
                losses[str_loss_dice] = self.loss_dice(pos_mask_pred,
                                                     pos_mask_targets,
                                                    pos_mask_weight)
#计算iou的二值交叉熵loss
                pos_mask_pred_1 = pos_mask_pred.detach().flatten(1)
                pos_mask_targets_1 = pos_mask_targets.detach().flatten(1)
                iou_labels = compute_mask_iou(pos_mask_pred_1, pos_mask_targets_1).unsqueeze(1)
                #print("-"*100)
                #print("pos_mask_targets", pos_mask_targets) #one-hot, float
                #print("pos_mask_pred.shape", pos_mask_pred.shape) #torch.Size([3, 200, 272])
                #print("iou_labels", iou_labels)
                #print("iou_labels.shape", iou_labels.shape) #torch.Size([3, 1])
                if iou_scores is not None:
                    pos_iou_scores = iou_scores.view(num_preds, -1)[bool_pos_inds]
                    #print("+"*100)
                    #print("pos_iou_scores.shape", pos_iou_scores.shape) #torch.Size([3, 1])
                    #print("bool_pos_inds", bool_pos_inds)
                    #print("bool_pos_inds.shape", bool_pos_inds.shape) #torch.Size([200])
                    #print("pos_iou_scores", pos_iou_scores)
                    
                    losses[str_loss_iou] = self.loss_iou(
                            pos_iou_scores,
                            iou_labels)
                
                
                if self.loss_rank is not None:
                    batch_size = mask_pred.size(0)
                    rank_target = mask_targets.new_full((batch_size, H, W),
                                                        self.ignore_label,
                                                        dtype=torch.long)
                    rank_inds = pos_inds.view(batch_size,
                                              -1).nonzero(as_tuple=False)
                    batch_mask_targets = mask_targets.view(
                        batch_size, -1, H, W).bool()
                    for i in range(batch_size):
                        curr_inds = (rank_inds[:, 0] == i)
                        curr_rank = rank_inds[:, 1][curr_inds]
                        for j in curr_rank:
                            rank_target[i][batch_mask_targets[i][j]] = j
                    losses['loss_rank'] = self.loss_rank(
                        mask_pred, rank_target, ignore_index=self.ignore_label)
            else:
                losses[str_loss_mask] = mask_pred.sum() * 0
                losses[str_loss_dice] = mask_pred.sum() * 0
                if iou_scores is not None:
                    losses[str_loss_iou] = mask_pred.sum() * 0
                if self.loss_rank is not None:
                    losses['loss_rank'] = mask_pred.sum() * 0

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
        label_weights = pos_mask.new_zeros((num_samples, self.num_classes))
        mask_targets = pos_mask.new_zeros(num_samples, H, W)
        #mask_weights = pos_mask.new_zeros(num_samples, H, W)
        mask_weights = pos_mask.new_zeros(num_samples)
        if num_pos > 0:
            labels[pos_inds] = pos_gt_labels
            
            
            #pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            #label_weights[pos_inds] = pos_weight
#修改权重，分类采用score，mask采用iou--------------------------------#            
            #pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight #cfg 1
            if flag == 1:
                pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
                label_weights[pos_inds] = pos_weight
            else:
                pos_score = pos_score.unsqueeze(1).expand([num_pos, self.num_classes])
                #label_weights[pos_inds] = pos_score * pos_score
#修改分类的权重，>=0.5之后就不再使用动态阈值                
                pos_score = pos_score.clamp(0, 0.5)
                   
        
                label_weights[pos_inds] = pos_score * pos_score * pos_score * pos_score * 12 
                #label_weights[pos_inds] = 1 # change
                
                '''
                print("head_get"+"="*100)
                import numpy as np
                np.set_printoptions(threshold=np.inf)
                lw = label_weights.detach().cpu().numpy()
                lb = labels.detach().cpu().numpy()
                
                print("label_weights", lw)
                print("labels", lb)
                '''
            
            pos_mask_targets = pos_gt_mask
            
            
#---------------------------------------#            
            
            
            mask_targets[pos_inds, ...] = pos_mask_targets
            
            if flag == 1:
                #mask_weights[pos_inds, ...] = 1
                mask_weights[pos_inds] = 1
            else:
                assert num_pos == pos_iou.shape[0]
                #pos_iou = pos_iou.unsqueeze(1).unsqueeze(1).expand([num_pos, H, W])
                #mask_weights[pos_inds, ...] = pos_iou
                #mask_weights[pos_inds] = pos_iou * pos_iou
                mask_weights[pos_inds] = pos_iou * pos_iou * pos_iou * pos_iou * 2.5
                #mask_weights[pos_inds] = 1 # change
                
            #mask_weights[pos_inds, ...] = 1

        if num_neg > 0:
            if flag == 1:
                label_weights[neg_inds] = 1.0
            else:
                label_weights[neg_inds] = 0.35 * 0.35
                #label_weights[neg_inds] = 1 # change
                
                
        if gt_sem_cls is not None and gt_sem_seg is not None:
            sem_labels = pos_mask.new_full((self.num_stuff_classes, ),
                                           self.num_classes,
                                           dtype=torch.long)
            sem_targets = pos_mask.new_zeros(self.num_stuff_classes, H, W)
            sem_weights = pos_mask.new_zeros(self.num_stuff_classes, H, W)
            sem_stuff_weights = torch.eye(
                self.num_stuff_classes, device=pos_mask.device)
            sem_thing_weights = pos_mask.new_zeros(
                (self.num_stuff_classes, self.num_thing_classes))
            sem_label_weights = torch.cat(
                [sem_thing_weights, sem_stuff_weights], dim=-1)
            if len(gt_sem_cls > 0):
                sem_inds = gt_sem_cls - self.num_thing_classes
                sem_inds = sem_inds.long()
                sem_labels[sem_inds] = gt_sem_cls.long()
                sem_targets[sem_inds] = gt_sem_seg
                sem_weights[sem_inds] = 1

            label_weights[:, self.num_thing_classes:] = 0
            labels = torch.cat([labels, sem_labels])
            label_weights = torch.cat([label_weights, sem_label_weights])
            mask_targets = torch.cat([mask_targets, sem_targets])
            mask_weights = torch.cat([mask_weights, sem_weights])

        return labels, label_weights, mask_targets, mask_weights

    def get_targets(self,
                    sampling_results,
                    gt_mask,
                    gt_labels,
                    rcnn_train_cfg,
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
        if gt_sem_seg is None:
            gt_sem_seg = [None] * num
            gt_sem_cls = [None] * num
    
#拿出pos score和iou
        pos_assigned_gt_inds_list = [res.pos_assigned_gt_inds for res in sampling_results]
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
        flag_list = [flag for i in range(num_image)]
        if gt_scores is None:
            pos_score_list = [None] * num
            pos_iou_list = [None] * num
        
        labels, label_weights, mask_targets, mask_weights = multi_apply(
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
            cfg=rcnn_train_cfg)
        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            mask_targets = torch.cat(mask_targets, 0)
            mask_weights = torch.cat(mask_weights, 0)
        return labels, label_weights, mask_targets, mask_weights

    def rescale_masks(self, masks_per_img, img_meta):
        h, w, _ = img_meta['img_shape']
        masks_per_img = F.interpolate(
            masks_per_img.unsqueeze(0).sigmoid(),
            size=img_meta['batch_input_shape'],
            mode='bilinear',
            align_corners=False)

        masks_per_img = masks_per_img[:, :, :h, :w]
        ori_shape = img_meta['ori_shape']
        seg_masks = F.interpolate(
            masks_per_img,
            size=ori_shape[:2],
            mode='bilinear',
            align_corners=False).squeeze(0)
        return seg_masks

    def get_seg_masks(self, masks_per_img, labels_per_img, scores_per_img,
                      test_cfg, img_meta):
        # resize mask predictions back
        
        seg_masks = self.rescale_masks(masks_per_img, img_meta)
        seg_masks = seg_masks > test_cfg.mask_thr        
        
        """
#可能无标签图片中没有大于0.5分类阈值的伪mask
        num = masks_per_img.shape[0]
        #print(num)
        
        #print("masks_per_img", masks_per_img, masks_per_img.shape)
        if num != 0:
            seg_masks = self.rescale_masks(masks_per_img, img_meta)
            seg_masks = seg_masks > test_cfg.mask_thr
        else:
            seg_masks = masks_per_img
        #scores_per_img, labels_per_img, seg_masks, keep_inds = mask_matrix_nms(seg_masks, labels_per_img, scores_per_img)
        #print("scores_per_img", scores_per_img, scores_per_img.shape)
        #print("labels_per_img", labels_per_img)
        #print("seg_masks", seg_masks, seg_masks.shape)
        """
     
        bbox_result, segm_result = self.segm2result(seg_masks, labels_per_img,
                                                    scores_per_img)
        return bbox_result, segm_result

    def segm2result(self, mask_preds, det_labels, cls_scores):
        num_classes = self.num_classes
        bbox_result = None
        segm_result = [[] for _ in range(num_classes)]
        mask_preds = mask_preds.cpu().numpy()
        det_labels = det_labels.cpu().numpy()
        cls_scores = cls_scores.cpu().numpy()
        num_ins = mask_preds.shape[0]
        # fake bboxes
        bboxes = np.zeros((num_ins, 5), dtype=np.float32)
        bboxes[:, -1] = cls_scores
        bbox_result = [bboxes[det_labels == i, :] for i in range(num_classes)]
        for idx in range(num_ins):
            segm_result[det_labels[idx]].append(mask_preds[idx])
        return bbox_result, segm_result
