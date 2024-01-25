_base_ = "base_weight_cx.py"
dataset_type = 'CocoDataset'
data_root = 'data/coco/'

data = dict(
    samples_per_gpu=3,
    workers_per_gpu=6,
    train=dict(
        sup=dict(
            type="CocoDataset",
            ann_file="data/coco/annotations/semi_supervised/instances_train2017.${fold}@${percent}.json",
            img_prefix="data/coco/train2017/",
        ),
        unsup=dict(
            type="CocoDataset",
            ann_file="data/coco/annotations/semi_supervised/instances_train2017.${fold}@${percent}-unlabeled.json",
            img_prefix="data/coco/train2017/",
        ),
    ),
    sampler=dict(
        train=dict(
            sample_ratio=[1, 2],
        )
    ),
)

fold = 1
percent = 10

#work_dir = "work_dirs/${cfg_name}/${percent}/${fold}"
work_dir="work_dirs/ssl/knet_gmm/10_percent_2_2"

# resume_from = None
resume_from = "work_dirs/ssl/knet_gmm/10_percent_2_2/latest.pth"

log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
    ],
)
