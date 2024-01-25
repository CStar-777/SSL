_base_ = "base_weight_cp.py"
dataset_type = "CocoDataset"
data_root = "data/coco/"

# ann_file = ['anno_file_1', 'anno_file_2']
# ConcatDataset继承于pytorch的ConcatDataset，
# 当配置文件中的anno_file存在多个值时，
# 会自动把包装成ConcatDataset，就是把多个小数据集合并

data = dict(
    samples_per_gpu=3,
    workers_per_gpu=6,
    train=dict(
        sup=dict(
            type="CocoDataset",
            ann_file="/data/Datasets_cx/data/coco/annotations/semi_supervised/instances_train2017.${fold}@${percent}.json",
            img_prefix="/data/Datasets_cx/data/coco/train2017/",
        ),
        unsup=dict(
            type="CocoDataset",
            ann_file="/data/Datasets_cx/data/coco/annotations/semi_supervised/instances_train2017.${fold}@${percent}-unlabeled.json",
            img_prefix="/data/Datasets_cx/data/coco/train2017/",
        ),
    ),
    sampler=dict(
        train=dict(
            sample_ratio=[1, 2],
        )
    ),
)
# '''

fold = 1
percent = 10

# work_dir = "work_dirs/${cfg_name}/${percent}/${fold}"
work_dir = "work_dirs/ssl/knet_class_cp_10"

# resume_from = None
# resume_from = "work_dirs/ssl/knet_class_cp_10/latest.pth"
resume_from = "work_dirs/ssl/knet_cp_gmm_10_2000/iter_136000.pth"
# resume_from = "work_dirs/ssl/knet_gmm/10_percent_2_2/iter_8000.pth"

log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
    ],
)
