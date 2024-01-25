mmdet_base = "../../thirdparty/mmdetection/configs/_base_"
_base_ = [
    '../_base_/schedules/schedule_1x.py',
    '../_base_/models/knet_s3_r50_fpn_weight_cs.py',
    f"{mmdet_base}/datasets/cityscapes_instance.py",
    '../_base_/default_runtime.py'
]

"""
model = dict(
    backbone=dict(
        norm_cfg=dict(requires_grad=False),
        norm_eval=True,
        style="caffe",
        init_cfg=dict(
            type="Pretrained", checkpoint="open-mmlab://detectron2/resnet50_caffe"
        ),
    )
)
"""
#img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

#！ 修改LoadAnnotations, collect
#!!!!!!!!!
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(
        type="Sequential",
        transforms=[
            dict(
                type="RandResize",
                img_scale=[(1333, 400), (1333, 1200)],
                #img_scale=[(1024, 512), (1024, 512)],
                multiscale_mode="range",
                keep_ratio=True,
            ),
            dict(type="RandFlip", flip_ratio=0.5),
        ],
        record=True,
    ),
    dict(type="Pad", size_divisor=32),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ExtraAttrs", tag="sup"),
    dict(type="DefaultFormatBundle"),
    dict(
        type="Collect",
        keys=["img", "gt_bboxes", "gt_labels", 'gt_masks'],
        meta_keys=(
            "filename",
            "ori_shape",
            "img_shape",
            "img_norm_cfg",
            "pad_shape",
            "scale_factor",
            "tag",
        ),
    ),
]

#！ 修改collect
#!!!!!!!!!
strong_pipeline = [
    dict(
        type="Sequential",
        transforms=[
            dict(
                type="RandResize",
                img_scale=[(1333, 400), (1333, 1200)],
                #img_scale=[(1024, 512), (1024, 512)],
                multiscale_mode="range",
                keep_ratio=True,
            ),
            dict(type="RandFlip", flip_ratio=0.5),
            dict(
                type="ShuffledSequential",
                transforms=[
                    dict(
                        type="OneOf",
                        transforms=[
                            dict(type=k)
                            for k in [
                                "Identity",
                                "AutoContrast",
                                "RandEqualize",
                                "RandSolarize",
                                "RandColor",
                                "RandContrast",
                                "RandBrightness",
                                "RandSharpness",
                                "RandPosterize",
                            ]
                        ],
                    ),
                    dict(
                        type="OneOf",
                        transforms=[
                            dict(type="RandTranslate", x=(-0.1, 0.1)),
                            dict(type="RandTranslate", y=(-0.1, 0.1)),
                            dict(type="RandRotate", angle=(-30, 30)),
                            [
                                dict(type="RandShear", x=(-30, 30)),
                                dict(type="RandShear", y=(-30, 30)),
                            ],
                        ],
                    ),
                    dict(
                        type="RandErase",
                        n_iterations=(1, 5),
                        size=[0, 0.2],
                        squared=True,
                    ),                    
                ],
            ),
        ],
        record=True,
    ),
    dict(type="Pad", size_divisor=32),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ExtraAttrs", tag="unsup_student"),
    dict(type="DefaultFormatBundle"),
    dict(
        type="Collect",
        keys=["img", "gt_bboxes", "gt_labels", 'gt_masks'],
        meta_keys=(
            "filename",
            "ori_shape",
            "img_shape",
            "img_norm_cfg",
            "pad_shape",
            "scale_factor",
            "tag",
            "transform_matrix",
        ),
    ),
]

#！ 修改collect
#!!!!!!!!!
weak_pipeline = [
    dict(
        type="Sequential",
        transforms=[
            dict(
                type="RandResize",
                img_scale=[(1333, 400), (1333, 1200)],
                #img_scale=[(1333, 400), (1333, 1200)],
                #img_scale=[(1024, 512), (1024, 512)],
                multiscale_mode="range",
                keep_ratio=True,
            ),
            dict(type="RandFlip", flip_ratio=0.5),
        ],
        record=True,
    ),
    dict(type="Pad", size_divisor=32),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ExtraAttrs", tag="unsup_teacher"),
    dict(type="DefaultFormatBundle"),
    dict(
        type="Collect",
        keys=["img", "gt_bboxes", "gt_labels", 'gt_masks'],
        meta_keys=(
            "filename",
            "ori_shape",
            "img_shape",
            "img_norm_cfg",
            "pad_shape",
            "scale_factor",
            "tag",
            "transform_matrix",
        ),
    ),
]

#！ 修改PseudoSamples, 并且去类中查看是否支持with_mask=True，PseudoSamples类有些疑惑，可以参考LoadAnnotations的源码来理解
#!!!!!!!!!
unsup_pipeline = [
    dict(type="LoadImageFromFile"),
    # dict(type="LoadAnnotations", with_bbox=True),
    # generate fake labels for data format compatibility
    dict(type="PseudoSamples", with_bbox=True, with_mask=True),
    dict(
        type="MultiBranch", unsup_student=strong_pipeline, unsup_teacher=weak_pipeline
    ),
]

test_pipeline = [
    
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]
dataset_type = 'CityscapesDataset'
data_root = 'data/cityscapes/'

data = dict(
    samples_per_gpu=None,
    workers_per_gpu=None,
    train=dict(
        _delete_=True,
        type="SemiDataset",
        sup=dict(
            type='RepeatDataset',
            times=8,
            dataset=dict(
                type=dataset_type,
                ann_file=data_root +
                'annotations/instancesonly_filtered_gtFine_train.json',
                img_prefix=data_root + 'leftImg8bit/train/',
                pipeline=train_pipeline,
            ),
        ),
        unsup=dict(
            type='RepeatDataset',
            times=8,
            dataset=dict(
                type=dataset_type,
                ann_file=data_root +
                'annotations/instancesonly_filtered_gtFine_train.json',
                img_prefix=data_root + 'leftImg8bit/train/',
                pipeline=unsup_pipeline,
                filter_empty_gt=False,
            ),
            #pipeline=unsup_pipeline,
        ),
    ),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline),
    sampler=dict(
        train=dict(
            type="SemiBalanceSampler",
            sample_ratio=[1, 4],
            by_prob=True,
            # at_least_one=True,
            epoch_length=7330,
        )
    ),
)

semi_wrapper = dict(
    type="SslKnet_weight_cs",
    model="${model}",
    train_cfg=dict(
        use_teacher_proposal=False,
        pseudo_label_initial_score_thr=0.3,
        pseudo_label_iou_thr=0.35,
        min_pseduo_box_size=0,
        unsup_weight=1.0,
    ),
    test_cfg=dict(inference_on="student"),
)


custom_hooks = [
    dict(type="WeightSummary"),
    dict(type="MeanTeacher", momentum=0.999, interval=1, warm_up=0),
]
#evaluation = dict(type="SubModulesDistEvalHook")
"""
custom_hooks = [
    dict(type="NumClassCheckHook"),
    dict(type="WeightSummary"),
    dict(type="MeanTeacher", momentum=0.999, interval=1, warm_up=0),
]
"""

evaluation = dict(type="SubModulesDistEvalHook", interval=3000)

#optimizer = dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0001)

#lr_config = dict(step=[120000, 160000])

lr_config = dict(
    policy='step',
    by_epoch=False,
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[7000])

runner = dict(_delete_=True, type="IterBasedRunner", max_iters=3000)

checkpoint_config = dict(by_epoch=False, interval=1000, max_keep_ckpts=5)

fp16 = dict(loss_scale="dynamic")


log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
    ])


