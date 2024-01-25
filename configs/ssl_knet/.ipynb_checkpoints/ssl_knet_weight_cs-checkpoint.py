_base_ = "base_weight_cs.py"
dataset_type = 'CityscapesDataset'
data_root = 'data/cityscapes/'
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        sup=dict(
            type='RepeatDataset',
            times=8,
            dataset=dict(
                type=dataset_type,
                ann_file=data_root +
                'annotations/instancesonly_filtered_gtFine_train.1@30.json',
                img_prefix=data_root + 'leftImg8bit/train/',
                #pipeline=train_pipeline,
            ),
        ),
        unsup=dict(
            type='RepeatDataset',
            times=8,
            dataset=dict(
                type=dataset_type,
                ann_file=data_root +
                'annotations/instancesonly_filtered_gtFine_train.1@30-unlabeled.json',
                img_prefix=data_root + 'leftImg8bit/train/',
                #pipeline=unsup_pipeline,
                #filter_empty_gt=False,
            ),
            #pipeline=unsup_pipeline,
        ),
    ),
    sampler=dict(
        train=dict(
            sample_ratio=[1, 3],
        )
    ),
)

fold = 1
percent = 1

#work_dir = "work_dirs/${cfg_name}/${percent}/${fold}"
#work_dir="work_dirs/ssl/weight_2_wo_score"
work_dir="work_dirs/ssl/cityscapes_bs_30/"

#resume_from = "work_dirs/ssl/ema_weight_0.35_0.5_1-2/iter_112000.pth"
#resume_from = "work_dirs/ssl/cityscapes_bs_30_1536/latest.pth"
#load_from = "work_dirs/ssl/ema_weight_0.35_0.5_1-2/iter_112000.pth"
#load_from = "work_dirs/ssl/ema_knet_1_1_0.35_0.3/iter_176000.pth"
#load_from = "work_dirs/ssl/mayue_test_ssl_f2_testV2/iter_22000.pth"
#load_from = "work_dirs/ssl/mayue_test_ssl_sc_test/iter_160000.pth"

'''
optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, momentum=0.9, weight_decay=0.0001)
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0001,
    weight_decay=0.05,
    eps=1e-8,
    betas=(0.9, 0.999))
'''

lr_config = dict(
    policy='step',
    step=[2650*18],
    warmup='linear',
    warmup_ratio=0.001, 
    warmup_iters=1000)

runner = dict(_delete_=True, type="IterBasedRunner", max_iters=3000*18)

log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
    ],
)