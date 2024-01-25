optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=0.05,
    paramwise_cfg=dict(custom_keys=dict(backbone=dict(lr_mult=0.25))))
optimizer_config = dict(grad_clip=dict(max_norm=1, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[160000, 200000])
runner = dict(type='IterBasedRunner', max_iters=220000)
num_stages = 3
num_proposals = 100
conv_kernel_size = 1
model = dict(
    type='SslKnet',
    model=dict(
        type='KNet',
        backbone=dict(
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            style='pytorch',
            init_cfg=dict(
                type='Pretrained', checkpoint='torchvision://resnet50')),
        neck=dict(
            type='FPN',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            start_level=0,
            add_extra_convs='on_input',
            num_outs=4),
        rpn_head=dict(
            type='ConvKernelHead',
            conv_kernel_size=1,
            feat_downsample_stride=2,
            feat_refine_stride=1,
            feat_refine=False,
            use_binary=True,
            num_loc_convs=1,
            num_seg_convs=1,
            conv_normal_init=True,
            localization_fpn=dict(
                type='SemanticFPNWrapper',
                in_channels=256,
                feat_channels=256,
                out_channels=256,
                start_level=0,
                end_level=3,
                upsample_times=2,
                positional_encoding=dict(
                    type='SinePositionalEncoding',
                    num_feats=128,
                    normalize=True),
                cat_coors=False,
                cat_coors_level=3,
                fuse_by_cat=False,
                return_list=False,
                num_aux_convs=1,
                norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)),
            num_proposals=100,
            proposal_feats_with_obj=True,
            xavier_init_kernel=False,
            kernel_init_std=1,
            num_cls_fcs=1,
            in_channels=256,
            num_classes=80,
            feat_transform_cfg=None,
            loss_seg=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_mask=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
            loss_dice=dict(type='DiceLoss', loss_weight=4.0)),
        roi_head=dict(
            type='KernelIterHead',
            num_stages=3,
            stage_loss_weights=[1, 1, 1],
            proposal_feature_channel=256,
            mask_head=[
                dict(
                    type='KernelUpdateHead',
                    num_classes=80,
                    num_ffn_fcs=2,
                    num_heads=8,
                    num_cls_fcs=1,
                    num_mask_fcs=1,
                    feedforward_channels=2048,
                    in_channels=256,
                    out_channels=256,
                    dropout=0.0,
                    mask_thr=0.5,
                    conv_kernel_size=1,
                    mask_upsample_stride=2,
                    ffn_act_cfg=dict(type='ReLU', inplace=True),
                    with_ffn=True,
                    feat_transform_cfg=dict(
                        conv_cfg=dict(type='Conv2d'), act_cfg=None),
                    kernel_updator_cfg=dict(
                        type='KernelUpdator',
                        in_channels=256,
                        feat_channels=256,
                        out_channels=256,
                        input_feat_shape=3,
                        act_cfg=dict(type='ReLU', inplace=True),
                        norm_cfg=dict(type='LN')),
                    loss_mask=dict(
                        type='CrossEntropyLoss',
                        use_sigmoid=True,
                        loss_weight=1.0),
                    loss_dice=dict(type='DiceLoss', loss_weight=4.0),
                    loss_cls=dict(
                        type='FocalLoss',
                        use_sigmoid=True,
                        gamma=2.0,
                        alpha=0.25,
                        loss_weight=2.0)),
                dict(
                    type='KernelUpdateHead',
                    num_classes=80,
                    num_ffn_fcs=2,
                    num_heads=8,
                    num_cls_fcs=1,
                    num_mask_fcs=1,
                    feedforward_channels=2048,
                    in_channels=256,
                    out_channels=256,
                    dropout=0.0,
                    mask_thr=0.5,
                    conv_kernel_size=1,
                    mask_upsample_stride=2,
                    ffn_act_cfg=dict(type='ReLU', inplace=True),
                    with_ffn=True,
                    feat_transform_cfg=dict(
                        conv_cfg=dict(type='Conv2d'), act_cfg=None),
                    kernel_updator_cfg=dict(
                        type='KernelUpdator',
                        in_channels=256,
                        feat_channels=256,
                        out_channels=256,
                        input_feat_shape=3,
                        act_cfg=dict(type='ReLU', inplace=True),
                        norm_cfg=dict(type='LN')),
                    loss_mask=dict(
                        type='CrossEntropyLoss',
                        use_sigmoid=True,
                        loss_weight=1.0),
                    loss_dice=dict(type='DiceLoss', loss_weight=4.0),
                    loss_cls=dict(
                        type='FocalLoss',
                        use_sigmoid=True,
                        gamma=2.0,
                        alpha=0.25,
                        loss_weight=2.0)),
                dict(
                    type='KernelUpdateHead',
                    num_classes=80,
                    num_ffn_fcs=2,
                    num_heads=8,
                    num_cls_fcs=1,
                    num_mask_fcs=1,
                    feedforward_channels=2048,
                    in_channels=256,
                    out_channels=256,
                    dropout=0.0,
                    mask_thr=0.5,
                    conv_kernel_size=1,
                    mask_upsample_stride=2,
                    ffn_act_cfg=dict(type='ReLU', inplace=True),
                    with_ffn=True,
                    feat_transform_cfg=dict(
                        conv_cfg=dict(type='Conv2d'), act_cfg=None),
                    kernel_updator_cfg=dict(
                        type='KernelUpdator',
                        in_channels=256,
                        feat_channels=256,
                        out_channels=256,
                        input_feat_shape=3,
                        act_cfg=dict(type='ReLU', inplace=True),
                        norm_cfg=dict(type='LN')),
                    loss_mask=dict(
                        type='CrossEntropyLoss',
                        use_sigmoid=True,
                        loss_weight=1.0),
                    loss_dice=dict(type='DiceLoss', loss_weight=4.0),
                    loss_cls=dict(
                        type='FocalLoss',
                        use_sigmoid=True,
                        gamma=2.0,
                        alpha=0.25,
                        loss_weight=2.0))
            ]),
        train_cfg=dict(
            rpn=dict(
                assigner=dict(
                    type='MaskHungarianAssigner',
                    cls_cost=dict(type='FocalLossCost_1', weight=2.0),
                    dice_cost=dict(type='DiceCost', weight=4.0, pred_act=True),
                    mask_cost=dict(type='MaskCost', weight=1.0,
                                   pred_act=True)),
                sampler=dict(type='MaskPseudoSampler'),
                pos_weight=1),
            rcnn=[
                dict(
                    assigner=dict(
                        type='MaskHungarianAssigner',
                        cls_cost=dict(type='FocalLossCost_1', weight=2.0),
                        dice_cost=dict(
                            type='DiceCost', weight=4.0, pred_act=True),
                        mask_cost=dict(
                            type='MaskCost', weight=1.0, pred_act=True)),
                    sampler=dict(type='MaskPseudoSampler'),
                    pos_weight=1),
                dict(
                    assigner=dict(
                        type='MaskHungarianAssigner',
                        cls_cost=dict(type='FocalLossCost_1', weight=2.0),
                        dice_cost=dict(
                            type='DiceCost', weight=4.0, pred_act=True),
                        mask_cost=dict(
                            type='MaskCost', weight=1.0, pred_act=True)),
                    sampler=dict(type='MaskPseudoSampler'),
                    pos_weight=1),
                dict(
                    assigner=dict(
                        type='MaskHungarianAssigner',
                        cls_cost=dict(type='FocalLossCost_1', weight=2.0),
                        dice_cost=dict(
                            type='DiceCost', weight=4.0, pred_act=True),
                        mask_cost=dict(
                            type='MaskCost', weight=1.0, pred_act=True)),
                    sampler=dict(type='MaskPseudoSampler'),
                    pos_weight=1)
            ]),
        test_cfg=dict(
            rpn=None,
            rcnn=dict(
                max_per_img=100,
                mask_thr=0.5,
                merge_stuff_thing=dict(
                    iou_thr=0.5, stuff_max_area=4096,
                    instance_score_thr=0.3)))),
    train_cfg=dict(
        use_teacher_proposal=False,
        pseudo_label_initial_score_thr=0.5,
        min_pseduo_box_size=0,
        unsup_weight=0.3),
    test_cfg=dict(inference_on='student'))
custom_imports = dict(
    imports=[
        'ssod.knet.det.knet', 'ssod.knet.det.kernel_head',
        'ssod.knet.det.kernel_iter_head', 'ssod.knet.det.kernel_update_head',
        'ssod.knet.det.semantic_fpn_wrapper', 'ssod.knet.kernel_updator',
        'ssod.knet.det.mask_hungarian_assigner',
        'ssod.knet.det.mask_pseudo_sampler'
    ],
    allow_failed_imports=False)
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='Sequential',
        transforms=[
            dict(
                type='RandResize',
                img_scale=[(1333, 400), (1333, 1200)],
                multiscale_mode='range',
                keep_ratio=True),
            dict(type='RandFlip', flip_ratio=0.5),
            dict(
                type='OneOf',
                transforms=[
                    dict(type='Identity'),
                    dict(type='AutoContrast'),
                    dict(type='RandEqualize'),
                    dict(type='RandSolarize'),
                    dict(type='RandColor'),
                    dict(type='RandContrast'),
                    dict(type='RandBrightness'),
                    dict(type='RandSharpness'),
                    dict(type='RandPosterize')
                ])
        ],
        record=True),
    dict(type='Pad', size_divisor=32),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='ExtraAttrs', tag='sup'),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'],
        meta_keys=('filename', 'ori_shape', 'img_shape', 'img_norm_cfg',
                   'pad_shape', 'scale_factor', 'tag'))
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=3,
    workers_per_gpu=3,
    train=dict(
        type='SemiDataset',
        sup=dict(
            type='CocoDataset',
            ann_file=
            'data/coco/annotations/semi_supervised/instances_train2017.1@10.json',
            img_prefix='data/coco/train2017/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
                dict(
                    type='Sequential',
                    transforms=[
                        dict(
                            type='RandResize',
                            img_scale=[(1333, 400), (1333, 1200)],
                            multiscale_mode='range',
                            keep_ratio=True),
                        dict(type='RandFlip', flip_ratio=0.5),
                        dict(
                            type='OneOf',
                            transforms=[
                                dict(type='Identity'),
                                dict(type='AutoContrast'),
                                dict(type='RandEqualize'),
                                dict(type='RandSolarize'),
                                dict(type='RandColor'),
                                dict(type='RandContrast'),
                                dict(type='RandBrightness'),
                                dict(type='RandSharpness'),
                                dict(type='RandPosterize')
                            ])
                    ],
                    record=True),
                dict(type='Pad', size_divisor=32),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='ExtraAttrs', tag='sup'),
                dict(type='DefaultFormatBundle'),
                dict(
                    type='Collect',
                    keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'],
                    meta_keys=('filename', 'ori_shape', 'img_shape',
                               'img_norm_cfg', 'pad_shape', 'scale_factor',
                               'tag'))
            ]),
        unsup=dict(
            type='CocoDataset',
            ann_file=
            'data/coco/annotations/semi_supervised/instances_train2017.1@10-unlabeled.json',
            img_prefix='data/coco/train2017/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='PseudoSamples', with_bbox=True, with_mask=True),
                dict(
                    type='MultiBranch',
                    unsup_student=[
                        dict(
                            type='Sequential',
                            transforms=[
                                dict(
                                    type='RandResize',
                                    img_scale=[(1333, 400), (1333, 1200)],
                                    multiscale_mode='range',
                                    keep_ratio=True),
                                dict(type='RandFlip', flip_ratio=0.5),
                                dict(
                                    type='ShuffledSequential',
                                    transforms=[
                                        dict(
                                            type='OneOf',
                                            transforms=[
                                                dict(type='Identity'),
                                                dict(type='AutoContrast'),
                                                dict(type='RandEqualize'),
                                                dict(type='RandSolarize'),
                                                dict(type='RandColor'),
                                                dict(type='RandContrast'),
                                                dict(type='RandBrightness'),
                                                dict(type='RandSharpness'),
                                                dict(type='RandPosterize')
                                            ]),
                                        dict(
                                            type='OneOf',
                                            transforms=[{
                                                'type': 'RandTranslate',
                                                'x': (-0.1, 0.1)
                                            }, {
                                                'type': 'RandTranslate',
                                                'y': (-0.1, 0.1)
                                            }, {
                                                'type': 'RandRotate',
                                                'angle': (-30, 30)
                                            },
                                                        [{
                                                            'type':
                                                            'RandShear',
                                                            'x': (-30, 30)
                                                        }, {
                                                            'type':
                                                            'RandShear',
                                                            'y': (-30, 30)
                                                        }]]),
                                        dict(
                                            type='RandErase',
                                            n_iterations=(1, 5),
                                            size=[0, 0.2],
                                            squared=True)
                                    ])
                            ],
                            record=True),
                        dict(type='Pad', size_divisor=32),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='ExtraAttrs', tag='unsup_student'),
                        dict(type='DefaultFormatBundle'),
                        dict(
                            type='Collect',
                            keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'],
                            meta_keys=('filename', 'ori_shape', 'img_shape',
                                       'img_norm_cfg', 'pad_shape',
                                       'scale_factor', 'tag',
                                       'transform_matrix'))
                    ],
                    unsup_teacher=[
                        dict(
                            type='Sequential',
                            transforms=[
                                dict(
                                    type='RandResize',
                                    img_scale=[(1333, 400), (1333, 1200)],
                                    multiscale_mode='range',
                                    keep_ratio=True),
                                dict(type='RandFlip', flip_ratio=0.5)
                            ],
                            record=True),
                        dict(type='Pad', size_divisor=32),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='ExtraAttrs', tag='unsup_teacher'),
                        dict(type='DefaultFormatBundle'),
                        dict(
                            type='Collect',
                            keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'],
                            meta_keys=('filename', 'ori_shape', 'img_shape',
                                       'img_norm_cfg', 'pad_shape',
                                       'scale_factor', 'tag',
                                       'transform_matrix'))
                    ])
            ],
            filter_empty_gt=False)),
    val=dict(
        type='CocoDataset',
        ann_file='data/coco/annotations/instances_val2017.json',
        img_prefix='data/coco/val2017/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='CocoDataset',
        ann_file='data/coco/annotations/instances_val2017.json',
        img_prefix='data/coco/val2017/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    sampler=dict(
        train=dict(
            type='SemiBalanceSampler',
            sample_ratio=[1, 2],
            by_prob=True,
            epoch_length=7330)))
evaluation = dict(
    metric=['segm'], type='SubModulesDistEvalHook', interval=8000)
checkpoint_config = dict(interval=2000, by_epoch=False, max_keep_ckpts=10)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = "work_dirs/mayue_iccv23/ema_knet_0.5_0.3_1-2/iter_110000.pth"
workflow = [('train', 1)]
strong_pipeline = [
    dict(
        type='Sequential',
        transforms=[
            dict(
                type='RandResize',
                img_scale=[(1333, 400), (1333, 1200)],
                multiscale_mode='range',
                keep_ratio=True),
            dict(type='RandFlip', flip_ratio=0.5),
            dict(
                type='ShuffledSequential',
                transforms=[
                    dict(
                        type='OneOf',
                        transforms=[
                            dict(type='Identity'),
                            dict(type='AutoContrast'),
                            dict(type='RandEqualize'),
                            dict(type='RandSolarize'),
                            dict(type='RandColor'),
                            dict(type='RandContrast'),
                            dict(type='RandBrightness'),
                            dict(type='RandSharpness'),
                            dict(type='RandPosterize')
                        ]),
                    dict(
                        type='OneOf',
                        transforms=[{
                            'type': 'RandTranslate',
                            'x': (-0.1, 0.1)
                        }, {
                            'type': 'RandTranslate',
                            'y': (-0.1, 0.1)
                        }, {
                            'type': 'RandRotate',
                            'angle': (-30, 30)
                        },
                                    [{
                                        'type': 'RandShear',
                                        'x': (-30, 30)
                                    }, {
                                        'type': 'RandShear',
                                        'y': (-30, 30)
                                    }]]),
                    dict(
                        type='RandErase',
                        n_iterations=(1, 5),
                        size=[0, 0.2],
                        squared=True)
                ])
        ],
        record=True),
    dict(type='Pad', size_divisor=32),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='ExtraAttrs', tag='unsup_student'),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'],
        meta_keys=('filename', 'ori_shape', 'img_shape', 'img_norm_cfg',
                   'pad_shape', 'scale_factor', 'tag', 'transform_matrix'))
]
weak_pipeline = [
    dict(
        type='Sequential',
        transforms=[
            dict(
                type='RandResize',
                img_scale=[(1333, 400), (1333, 1200)],
                multiscale_mode='range',
                keep_ratio=True),
            dict(type='RandFlip', flip_ratio=0.5)
        ],
        record=True),
    dict(type='Pad', size_divisor=32),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='ExtraAttrs', tag='unsup_teacher'),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'],
        meta_keys=('filename', 'ori_shape', 'img_shape', 'img_norm_cfg',
                   'pad_shape', 'scale_factor', 'tag', 'transform_matrix'))
]
unsup_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='PseudoSamples', with_bbox=True, with_mask=True),
    dict(
        type='MultiBranch',
        unsup_student=[
            dict(
                type='Sequential',
                transforms=[
                    dict(
                        type='RandResize',
                        img_scale=[(1333, 400), (1333, 1200)],
                        multiscale_mode='range',
                        keep_ratio=True),
                    dict(type='RandFlip', flip_ratio=0.5),
                    dict(
                        type='ShuffledSequential',
                        transforms=[
                            dict(
                                type='OneOf',
                                transforms=[
                                    dict(type='Identity'),
                                    dict(type='AutoContrast'),
                                    dict(type='RandEqualize'),
                                    dict(type='RandSolarize'),
                                    dict(type='RandColor'),
                                    dict(type='RandContrast'),
                                    dict(type='RandBrightness'),
                                    dict(type='RandSharpness'),
                                    dict(type='RandPosterize')
                                ]),
                            dict(
                                type='OneOf',
                                transforms=[{
                                    'type': 'RandTranslate',
                                    'x': (-0.1, 0.1)
                                }, {
                                    'type': 'RandTranslate',
                                    'y': (-0.1, 0.1)
                                }, {
                                    'type': 'RandRotate',
                                    'angle': (-30, 30)
                                },
                                            [{
                                                'type': 'RandShear',
                                                'x': (-30, 30)
                                            }, {
                                                'type': 'RandShear',
                                                'y': (-30, 30)
                                            }]]),
                            dict(
                                type='RandErase',
                                n_iterations=(1, 5),
                                size=[0, 0.2],
                                squared=True)
                        ])
                ],
                record=True),
            dict(type='Pad', size_divisor=32),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ExtraAttrs', tag='unsup_student'),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'],
                meta_keys=('filename', 'ori_shape', 'img_shape',
                           'img_norm_cfg', 'pad_shape', 'scale_factor', 'tag',
                           'transform_matrix'))
        ],
        unsup_teacher=[
            dict(
                type='Sequential',
                transforms=[
                    dict(
                        type='RandResize',
                        img_scale=[(1333, 400), (1333, 1200)],
                        multiscale_mode='range',
                        keep_ratio=True),
                    dict(type='RandFlip', flip_ratio=0.5)
                ],
                record=True),
            dict(type='Pad', size_divisor=32),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ExtraAttrs', tag='unsup_teacher'),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'],
                meta_keys=('filename', 'ori_shape', 'img_shape',
                           'img_norm_cfg', 'pad_shape', 'scale_factor', 'tag',
                           'transform_matrix'))
        ])
]
custom_hooks = [
    dict(type='WeightSummary'),
    dict(type='MeanTeacher', momentum=0.999, interval=1, warm_up=0)
]
fp16 = dict(loss_scale='dynamic')
fold = 1
percent = 10
work_dir = 'work_dirs/mayue_iccv23/ema_knet_0.5_0.3_1-2'
cfg_name = 'ssl_knet_73'
gpu_ids = range(0, 4)
