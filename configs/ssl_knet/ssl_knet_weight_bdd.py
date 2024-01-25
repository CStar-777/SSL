_base_ = "base_weight_bdd.py"
data_root = 'data/BDD100K/'
data = dict(
    samples_per_gpu=3,
    workers_per_gpu=6,
    train=dict(
        sup=dict(
            type="CocoDataset",
            ann_file=data_root + 'bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_det_coco_train.${fold}@${percent}.json',
            img_prefix='data/BDD100K/bdd100k/bdd100k/images/100k/train',
        ),
        unsup=dict(
            type="CocoDataset",
            ann_file=data_root + 'bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_det_coco_train.${fold}@${percent}-unlabeled.json',
            img_prefix='data/BDD100K/bdd100k/bdd100k/images/100k/train',
        ),
    ),
    val=dict(
        type="CocoDataset",
        ann_file='data/BDD100K/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_det_coco_val.json',
        img_prefix='data/BDD100K/bdd100k/bdd100k/images/100k/val'),
    test=dict(
        type="CocoDataset",
        ann_file='data/BDD100K/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_det_coco_val.json',
        img_prefix='data/BDD100K/bdd100k/bdd100k/images/100k/val'),
    
    sampler=dict(
        train=dict(
            sample_ratio=[1, 2],
        )
    ),
)

fold = 1
percent = 1

#work_dir = "work_dirs/${cfg_name}/${percent}/${fold}"
work_dir="work_dirs/ssl/BDD_test"
#work_dir="work_dirs/test"

#resume_from = "work_dirs/ema_weight_2_3/iter_80000.pth"
#resume_from = "work_dirs/ssl/knet_2_percent_test/iter_80000.pth"
# resume_from = None

log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
    ],
)
