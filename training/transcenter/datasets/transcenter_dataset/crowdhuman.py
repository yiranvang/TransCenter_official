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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
try:
  from .generic_dataset import GenericDataset
except:
  from generic_dataset import GenericDataset

class CrowdHuman(GenericDataset):
  num_classes = 1
  num_joints = 17
  default_resolution = [640, 1088]
  max_objs = 300
  class_name = ['person']
  cat_ids = {1: 1}

  def __init__(self, opt, split):
    super(CrowdHuman, self).__init__()
    data_dir = opt.data_dir
    img_dir = os.path.join(
      data_dir, 'Images')
    ann_path = os.path.join(data_dir, 'annotations',
        '{}.json').format(split)


    print('==> initializing CrowdHuman {} data.'.format(split))

    self.images = None
    # load image list and coco
    super(CrowdHuman, self).__init__(opt, split, ann_path, img_dir)
    self.sf = 0.3

    self.num_samples = len(self.images)

    print('Loaded {} {} samples'.format(split, self.num_samples))

  def _to_float(self, x):
    return float("{:.2f}".format(x))

  def _save_results(self, records, fpath):
    with open(fpath,'w') as fid:
      for record in records:
        line = json.dumps(record)+'\n'
        fid.write(line)
    return fpath

  def convert_eval_format(self, all_bboxes):
    detections = []
    person_id = 1
    for image_id in all_bboxes:
      if type(all_bboxes[image_id]) != type({}):
        # newest format
        dtboxes = []
        for j in range(len(all_bboxes[image_id])):
          item = all_bboxes[image_id][j]
          if item['class'] != person_id:
            continue
          bbox = item['bbox']
          bbox[2] -= bbox[0]
          bbox[3] -= bbox[1]
          bbox_out  = list(map(self._to_float, bbox[0:4]))
          detection = {
              "tag": 1,
              "box": bbox_out,
              "score": float("{:.2f}".format(item['score']))
          }
          dtboxes.append(detection)
      img_info = self.coco.loadImgs(ids=[image_id])[0]
      file_name = img_info['file_name']
      detections.append({'ID': file_name[:-4], 'dtboxes': dtboxes})
    return detections

  def __len__(self):
    return self.num_samples

  def save_results(self, results, save_dir):
    self._save_results(self.convert_eval_format(results),
                       '{}/results_crowdhuman.odgt'.format(save_dir))
  def run_eval(self, results, save_dir):
    self.save_results(results, save_dir)
    try:
      os.system('python tools/crowdhuman_eval/demo.py ' + \
                '../data/crowdhuman/annotation_val.odgt ' + \
                '{}/results_crowdhuman.odgt'.format(save_dir))
    except:
      print('Crowdhuman evaluation not setup!')