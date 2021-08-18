##
## TransCenter: Transformers with Dense Queries for Multiple-Object Tracking
## Copyright Inria
## Year 2021
## Contact : yihong.xu@inria.fr
##
## The software TransCenter is provided "as is", for reproducibility purposes only.
## The user is not authorized to distribute the software TransCenter, modified or not.
## The user expressly undertakes not to remove, or modify, in any manner, the intellectual property notice attached to the software TransCenter.
## code modified from
## (1) Deformable-DETR, https://github.com/fundamentalvision/Deformable-DETR, distributed under Apache License 2.0 2020 fundamentalvision.
## (2) tracking_wo_bnw, https://github.com/phil-bergmann/tracking_wo_bnw, distributed under GNU General Public License v3.0 2020 Philipp Bergmann, Tim Meinhardt.
## (3) CenterTrack, https://github.com/xingyizhou/CenterTrack, distributed under MIT Licence 2020 Xingyi Zhou.
## (4) DCNv2, https://github.com/CharlesShang/DCNv2, distributed under BSD 3-Clause 2019 Charles Shang.
## (5) correlation_package, https://github.com/NVIDIA/flownet2-pytorch/tree/master/networks/correlation_package, Apache License, Version 2.0 2017 NVIDIA CORPORATION.
## (6) pytorch-liteflownet, https://github.com/sniklaus/pytorch-liteflownet, GNU General Public License v3.0 Simon Niklaus.
## (7) LiteFlowNet, https://github.com/twhui/LiteFlowNet, Copyright (c) 2018 Tak-Wai Hui.
## Below you can find the License associated to this LiteFlowNet software:
## 
##     This software and associated documentation files (the "Software"), and the research paper
##     (LiteFlowNet: A Lightweight Convolutional Neural Network for Optical Flow Estimation) including
##     but not limited to the figures, and tables (the "Paper") are provided for academic research purposes
##     only and without any warranty. Any commercial use requires my consent. When using any parts
##     of the Software or the Paper in your work, please cite the following paper:
## 
##     @InProceedings{hui18liteflownet,
##     author = {Tak-Wai Hui and Xiaoou Tang and Chen Change Loy},
##     title = {LiteFlowNet: A Lightweight Convolutional Neural Network for Optical Flow Estimation},
##     booktitle  = {Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
##     year = {2018},
##     pages = {8981--8989},
##     }
## 
##     The above copyright notice and this permission notice shall be included in all copies or
##     substantial portions of the Software.


"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable
import torch
import util.misc as utils
from datasets.coco_eval import CocoEvaluator


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, adaptive_clip: bool=False):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50

    for ret in metric_logger.log_every(data_loader, print_freq, header):
        samples = utils.NestedTensor(ret['image'], ret['pad_mask'])
        samples = samples.to(device)
        pre_samples = utils.NestedTensor(ret['pre_img'], ret['pre_pad_mask'])
        pre_hm = ret['pre_hm'].to(device)
        pre_samples = pre_samples.to(device)

        targets = {k: v.to(device) for k, v in ret.items() if
                   k != 'orig_image' and k != 'image' and 'pad_mask' not in k and 'pre_img' not in k}

        outputs = model(samples, pre_samples=pre_samples, pre_hm=pre_hm)
        loss_dict = criterion(outputs, targets)

        weight_dict = criterion.weight_dict

        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        assert len(weight_dict.keys()) == len(loss_dict_reduced.keys())

        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()

        losses.backward()

        if adaptive_clip:
            if max_norm > 0:
                utils.clip_grad_norm(model.parameters())
                grad_total_norm = utils.get_total_grad_norm(model.parameters())
            else:
                grad_total_norm = utils.get_total_grad_norm(model.parameters())
        else:
            if max_norm > 0:
                grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            else:
                grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)


        optimizer.step()

        # torch.cuda.empty_cache()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # set max Dets to 300
    coco_evaluator.coco_eval[iou_types[0]].params.maxDets = [300, 300, 300]

    for ret in metric_logger.log_every(data_loader, 50, header):
        samples = utils.NestedTensor(ret['image'], ret['pad_mask'])
        samples = samples.to(device)
        pre_samples = utils.NestedTensor(ret['pre_img'], ret['pre_pad_mask'])
        pre_hm = ret['pre_hm'].to(device)
        pre_samples = pre_samples.to(device)

        targets = {k: v.to(device) for k, v in ret.items() if k != 'orig_image' and k != 'image' and 'pad_mask' not in k and 'pre_img' not in k}

        outputs = model(samples, pre_samples=pre_samples, pre_hm=pre_hm)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)

        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)

        results = postprocessors['bbox'](outputs, targets['orig_size'], filter_score=False)
        res = {img_id.item(): output for img_id, output in zip(targets['image_id'], results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
    return stats, coco_evaluator
