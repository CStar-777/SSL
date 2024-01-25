from mmcv.parallel import is_module_wrapper
from mmcv.runner.hooks import HOOKS, Hook
from bisect import bisect_right
from ..logger import log_every_n
from mmdet.datasets.coco import *

import json
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.cuda.amp import autocast, GradScaler
from collections import Counter
import os
import contextlib

from sklearn.metrics import *
from copy import deepcopy

# undset_path = "data/coco/annotations/semi_supervised/instances_train2017.1@10-unlabeled.json"
# coco = COCO(undset_path)
# ulb_dset = len(coco.imgs)


@HOOKS.register_module()
class CurriculumPseudoLabel(Hook):
    def __init__(
        self,
        ulb_dset_len = 106459
    ):
        self.classwise_acc = torch.zeros((len(CocoDataset.CLASSES),)).cuda()
        self.selected_label = (torch.ones((ulb_dset_len,), dtype=torch.long, ) * -1).cuda()

    def before_train_iter(self, runner):
        """Update classwise_acc every self.interval iterations.
        
        """
        curr_step = runner.iter
        if curr_step % self.interval != 0:
            return
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        
        runner.log_buffer.output["class_acc"] = self.classwise_acc
        self.acc_update(model, classwise_acc)

    def after_train_iter(self, runner):
        curr_step = runner.iter
        
        
        self.class_acc = 
        
        self.momentum = 1 - (1 - self.momentum) * self.decay_factor ** bisect_right(
            self.decay_intervals, curr_step
        )

    def acc_update():
        
