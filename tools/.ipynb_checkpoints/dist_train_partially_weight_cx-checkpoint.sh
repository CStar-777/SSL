#!/usr/bin/env bash
set -x

TYPE=$1
FOLD=$2
PERCENT=$3
GPUS=$4
PORT=${PORT:-29000}


PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

if [[ ${TYPE} == 'baseline' ]]; then
    python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
        $(dirname "$0")/train.py configs/baseline/faster_rcnn_r50_caffe_fpn_coco_partial_180k.py --launcher pytorch \
        --cfg-options fold=${FOLD} percent=${PERCENT} ${@:5}
else
    CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
        $(dirname "$0")/train.py configs/ssl_knet/ssl_knet_weight_gmm.py --launcher pytorch \
        --cfg-options fold=${FOLD} percent=${PERCENT} ${@:5} >> train-gmm_10_2_2.out &
fi