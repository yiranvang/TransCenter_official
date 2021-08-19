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

import os
import numpy as np
import json
import cv2

DATA_PATH = 'CrowdHumanPath'
OUT_PATH = DATA_PATH + 'annotations/'
SPLITS = ['val', 'train']
DEBUG = False


def load_func(fpath):
    print('fpath', fpath)
    assert os.path.exists(fpath)
    with open(fpath, 'r') as fid:
        lines = fid.readlines()
    records = [json.loads(line.strip('\n')) for line in lines]
    return records


if __name__ == '__main__':
    import copy
    def bb_IoU(boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
        # return the intersection over union value
        return iou


    if not os.path.exists(OUT_PATH):
        os.mkdir(OUT_PATH)
    for split in SPLITS:
        data_path = DATA_PATH + split
        out_path = OUT_PATH + '{}.json'.format(split)
        out = {'images': [], 'annotations': [],
               'categories': [{'id': 1, 'name': 'person'}]}
        ann_path = DATA_PATH + '/annotation_{}.odgt'.format(split)
        anns_data = load_func(ann_path)
        image_cnt = 0
        ann_cnt = 0
        video_cnt = 0
        trackid_cnt = 0
        for ann_data in anns_data:
            # if (image_cnt >= 3750 and split == 'train') or \
            #     (image_cnt >= 1092 and split == 'val'):
            #     break
            image_cnt += 1
            image_info = {'file_name': '{}.jpg'.format(ann_data['ID']),
                          'id': image_cnt}
            out['images'].append(image_info)
            if split != 'test':
                anns = ann_data['gtboxes']
                for i in range(len(anns)):

                    if anns[i]['tag'] == 'person':
                        full_box = np.array(anns[i]['fbox'], dtype=np.float32).copy()
                        full_box[2:] = full_box[:2] + full_box[2:]
                        v_box = np.array(anns[i]['vbox'], dtype=np.float32).copy()
                        v_box[2:] = v_box[:2] + v_box[2:]
                        vis_level = bb_IoU(full_box, v_box)
                        # print("vis level:", vis_level)
                        iscrowd = ('extra' in anns[i] and
                                   'ignore' in anns[i]['extra'] and
                                   anns[i]['extra']['ignore'] == 1) or vis_level < 0.1
                        trackid_cnt += 1
                        ann_cnt += 1
                        ann = {'id': ann_cnt,
                               'category_id': 1,
                               'image_id': image_cnt,
                               'bbox_vis': anns[i]['vbox'],
                               'bbox': anns[i]['fbox'],
                               'iscrowd': int(iscrowd),
                               'area': anns[i]['fbox'][2]*anns[i]['fbox'][3],
                               'track_id': trackid_cnt}
                        out['annotations'].append(ann)
        print('loaded {} for {} images and {} samples'.format(
            split, len(out['images']), len(out['annotations'])))
        json.dump(out, open(out_path, 'w'))
