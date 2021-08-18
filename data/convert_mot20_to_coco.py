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

# Use the same script for MOT16
# DATA_PATH = '../../data/mot16/'
DATA_PATH = '/scratch/scorpio/yixu/rawdata/MOT20/'
OUT_PATH = DATA_PATH + 'annotations/'
SPLITS = ['train_half', 'val_half', 'train', 'test']
HALF_VIDEO = True
CREATE_SPLITTED_ANN = True
CREATE_SPLITTED_DET = True

if __name__ == '__main__':
  for split in SPLITS:
    data_path = DATA_PATH + 'train'
    if split == "test":
      data_path = DATA_PATH + 'test'

    out_path = OUT_PATH + '{}.json'.format(split)
    out = {'images': [], 'annotations': [], 
           'categories': [{'id': 1, 'name': 'pedestrian'}],
           'videos': []}
    seqs = os.listdir(data_path)
    image_cnt = 0
    ann_cnt = 0
    video_cnt = 0
    for seq in sorted(seqs):
      if '.DS_Store' in seq:
        continue
      if 'MOT20' not in DATA_PATH:
        continue
      video_cnt += 1
      out['videos'].append({
        'id': video_cnt,
        'file_name': seq})
      seq_path = '{}/{}/'.format(data_path, seq)
      img_path = seq_path + 'img1/'
      ann_path = seq_path + 'gt/gt.txt'
      images = os.listdir(img_path)
      num_images = len([image for image in images if 'jpg' in image])
      if HALF_VIDEO and ('half' in split):
        image_range = [0, num_images // 2] if 'train' in split else \
          [int(0.75*num_images+0.5), num_images - 1]
      else:
        image_range = [0, num_images - 1]
      for i in range(num_images):
        if (i < image_range[0] or i > image_range[1]):
          continue
        image_info = {'file_name': '{}/img1/{:06d}.jpg'.format(seq, i + 1),
                      'id': image_cnt + i + 1,
                      'frame_id': i + 1 - image_range[0],
                      'prev_image_id': image_cnt + i if i > 0 else -1,
                      'next_image_id': \
                        image_cnt + i + 2 if i < num_images - 1 else -1,
                      'video_id': video_cnt}
        out['images'].append(image_info)
      print('{}: {} images'.format(seq, num_images))
      if split != 'test':
        det_path = seq_path + 'det/det.txt'
        anns = np.loadtxt(ann_path, dtype=np.float32, delimiter=',')
        dets = np.loadtxt(det_path, dtype=np.float32, delimiter=',')
        if CREATE_SPLITTED_ANN and ('half' in split):
          anns_out = np.array([anns[i] for i in range(anns.shape[0]) if \
            int(anns[i][0]) - 1 >= image_range[0] and \
            int(anns[i][0]) - 1 <= image_range[1]], np.float32)
          anns_out[:, 0] -= image_range[0]
          gt_out = seq_path + '/gt/gt_{}.txt'.format(split)
          fout = open(gt_out, 'w')
          for o in anns_out:
            fout.write(
              '{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:.6f}\n'.format(
              int(o[0]),int(o[1]),int(o[2]),int(o[3]),int(o[4]),int(o[5]),
              int(o[6]),int(o[7]),o[8]))
          fout.close()
        if CREATE_SPLITTED_DET and ('half' in split):
          dets_out = np.array([dets[i] for i in range(dets.shape[0]) if \
            int(dets[i][0]) - 1 >= image_range[0] and \
            int(dets[i][0]) - 1 <= image_range[1]], np.float32)
          dets_out[:, 0] -= image_range[0]
          det_out = seq_path + '/det/det_{}.txt'.format(split)
          dout = open(det_out, 'w')
          for o in dets_out:
            dout.write(
              '{:d},{:d},{:.1f},{:.1f},{:.1f},{:.1f},{:.6f}\n'.format(
              int(o[0]),int(o[1]),float(o[2]),float(o[3]),float(o[4]),float(o[5]),
              float(o[6])))
          dout.close()

        print(' {} ann images'.format(int(anns[:, 0].max())))
        for i in range(anns.shape[0]):
          frame_id = int(anns[i][0])
          if (frame_id - 1 < image_range[0] or frame_id - 1> image_range[1]):
            continue
          track_id = int(anns[i][1])
          cat_id = int(anns[i][7])
          ann_cnt += 1
          iscrowd = 0
          if not ('15' in DATA_PATH):
            if not (float(anns[i][8]) >= 0.1):
              iscrowd = 1
            if not (int(anns[i][6]) == 1):
              continue
            if (int(anns[i][7]) in [3, 4, 5, 9, 10, 11]): # Non-person
              continue
            if (int(anns[i][7]) in [2, 7, 8, 12, 6]): # Ignored person
              category_id = -1
            else:
              category_id = 1
          else:
            category_id = 1
          ann = {'id': ann_cnt,
                 'iscrowd': int(iscrowd),
                 'category_id': category_id,
                 'image_id': image_cnt + frame_id,
                 'track_id': track_id,
                 'bbox': anns[i][2:6].tolist(),
                 'area': float(anns[i][4]*anns[i][5]),
                 'conf': float(anns[i][6])}
          out['annotations'].append(ann)
      image_cnt += num_images
    print('loaded {} for {} images and {} samples'.format(
      split, len(out['images']), len(out['annotations'])))
    json.dump(out, open(out_path, 'w'))
        
        

