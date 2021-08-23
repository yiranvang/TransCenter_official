## TransCenter: Transformers with Dense Queries for Multiple-Object Tracking
## Copyright Inria
## Year 2021
## Contact : yihong.xu@inria.fr
##
## TransCenter is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.

## TransCenter is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program, TransCenter.  If not, see <http://www.gnu.org/licenses/> and the LICENSE file.
##
##
## TransCenter has code derived from
## (1) 2020 fundamentalvision.(Apache License 2.0: https://github.com/fundamentalvision/Deformable-DETR)
## (2) 2020 Philipp Bergmann, Tim Meinhardt. (GNU General Public License v3.0 Licence: https://github.com/phil-bergmann/tracking_wo_bnw)
## (3) 2020 Facebook. (Apache License Version 2.0: https://github.com/facebookresearch/detr/)
## (4) 2020 Xingyi Zhou.(MIT License: https://github.com/xingyizhou/CenterTrack)
##
## TransCenter uses packages from
## (1) 2019 Charles Shang. (BSD 3-Clause Licence: https://github.com/CharlesShang/DCNv2)
## (2) 2017 NVIDIA CORPORATION. (Apache License, Version 2.0: https://github.com/NVIDIA/flownet2-pytorch/tree/master/networks/correlation_package)
## (3) 2019 Simon Niklaus. (GNU General Public License v3.0: https://github.com/sniklaus/pytorch-liteflownet)
## (4) 2018 Tak-Wai Hui. (Copyright (c), see details in the LICENSE file: https://github.com/twhui/LiteFlowNet)
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from scipy.optimize import linear_sum_assignment
import cv2
from PIL import Image, ImageDraw, ImageFont
from util.tracker_util import bbox_overlaps, warp_pos

from torchvision.ops.boxes import clip_boxes_to_image, nms
import torchvision
import math
from util.misc import get_flow, NestedTensor, gaussian_radius, affine_transform, draw_umich_gaussian, soft_nms_pytorch
import os
import lap


class Tracker:
    """The main tracking file, here is where magic happens."""
    # only track pedestrian
    cl = 1

    def __init__(self, obj_detect, reid_network, flownet, tracker_cfg, postprocessor=None, main_args=None):
        self.obj_detect = obj_detect
        self.reid_network = reid_network
        self.detection_nms_thresh = tracker_cfg['detection_nms_thresh']
        self.public_detections = tracker_cfg['public_detections']
        self.inactive_patience = tracker_cfg['inactive_patience']
        self.do_reid = tracker_cfg['do_reid']
        self.max_features_num = tracker_cfg['max_features_num']
        self.reid_sim_threshold = tracker_cfg['reid_sim_threshold']
        self.reid_iou_threshold = tracker_cfg['reid_iou_threshold']
        self.do_align = tracker_cfg['do_align']
        self.motion_model_cfg = tracker_cfg['motion_model']
        self.postprocessor = postprocessor
        self.main_args = main_args

        # self.tracks = []
        self.inactive_tracks = []
        self.track_num = 0
        self.im_index = 0
        self.results = {}

        #
        self.LiteFlowNet = flownet
        self.img_features = None
        self.encoder_pos_encoding = None
        self.transforms = transforms.ToTensor()
        self.last_image = None
        self.pre_sample = None
        self.sample = None
        self.pre_img_features = None
        self.pre_encoder_pos_encoding = None
        self.flow = None

    def reset(self, hard=True):
        self.tracks = []
        self.inactive_tracks = []
        self.last_image = None
        self.pre_sample = None
        self.sample = None
        self.pre_img_features = None
        self.pre_encoder_pos_encoding = None
        self.flow = None
        if hard:
            self.track_num = 0
            self.results = {}
            self.im_index = 0

    def linear_assignment(self, cost_matrix, thresh):
        if cost_matrix.size == 0:
            return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
        matches, unmatched_a, unmatched_b = [], [], []
        cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
        for ix, mx in enumerate(x):
            if mx >= 0:
                matches.append([ix, mx])
        unmatched_a = np.where(x < 0)[0]
        unmatched_b = np.where(y < 0)[0]
        matches = np.asarray(matches)
        return matches, unmatched_a, unmatched_b

    def tracks_to_inactive(self, tracks):
        self.tracks = [t for t in self.tracks if t not in tracks]
        for t in tracks:
            t.pos = t.last_pos[-1]
        self.inactive_tracks += tracks

    def add(self, new_det_pos, new_det_scores, new_det_features):
        """Initializes new Track objects and saves them."""
        num_new = new_det_pos.size(0)
        for i in range(num_new):
            self.tracks.append(Track(
                new_det_pos[i].view(1, -1),
                new_det_scores[i],
                self.track_num + i,
                new_det_features[i].view(1, -1),
                self.inactive_patience,
                self.max_features_num,
                self.motion_model_cfg['n_steps'] if self.motion_model_cfg['n_steps'] > 0 else 1
            ))
        self.track_num += num_new

    def tracks_dets_matching_tracking(self, raw_dets, raw_scores, det_pre_cts, img_pil=None, img_draw=None,
                                      pre_image_pil=None, pre_img_draw=None, blob=None):
        """
        raw_dets and raw_scores are clean (only ped class and filtered by a threshold
        """
        #
        pos = self.get_pos().clone()  # ? you are not in the same image plane
        # iou matching #
        assert pos.nelement() > 0
        if raw_dets.nelement() > 0:
            assert raw_dets.shape[0] == det_pre_cts.shape[0]
            # warped_pos = warp_pos(pos.cuda(), self.flow)
            # print(warped_pos) xyxy 0123
            raw_dets_w = raw_dets[:, [2]] - raw_dets[:, [0]]
            raw_dets_h = raw_dets[:, [3]] - raw_dets[:, [1]]

            warped_raw_dets = torch.cat([det_pre_cts[:, [0]] - 0.5 * raw_dets_w,
                                         det_pre_cts[:, [1]] - 0.5 * raw_dets_h,
                                         det_pre_cts[:, [0]] + 0.5 * raw_dets_w,
                                         det_pre_cts[:, [1]] + 0.5 * raw_dets_h], dim=1)

            # #todo check warp?
            # matching with IOU
            iou_dist = 1 - bbox_overlaps(pos.cuda(), warped_raw_dets.cuda())
            matches, u_track, u_detection = self.linear_assignment(iou_dist.cpu().numpy(), thresh=0.5)

            if matches.shape[0] == 0:  # nothing matched
                matches_dict = dict()
            else:
                matches_dict = dict(zip(matches[:, 0], matches[:, 1]))

            pos_birth = raw_dets[u_detection, :]
            scores_birth = raw_scores[u_detection]
        else:
            # no detection, kill all
            u_track = list(range(len(self.tracks)))
            matches_dict = {}
            pos_birth = torch.zeros(0, 4).cuda()
            scores_birth = torch.zeros(0).cuda()

        s = []
        new_pos = []
        for i in range(len(self.tracks) - 1, -1, -1):
            t = self.tracks[i]
            if i in u_track or raw_scores[matches_dict[i]] < self.main_args.track_thresh:
                self.tracks_to_inactive([t])
            else:
                s.append(raw_scores[matches_dict[i]])
                # t.prev_pos = t.pos
                t.pos = raw_dets[matches_dict[i]].view(1, -1).cuda()
                new_pos.append(t.pos.clone())
        try:
            return torch.Tensor(s[::-1]).cuda(), torch.cat(new_pos[::-1], dim=0), None, [pos_birth, scores_birth]

        except:
            return torch.zeros(0).cuda(), torch.zeros(0, 4).cuda(), torch.zeros(0).cuda(), [pos_birth, scores_birth]

    def detect_tracking(self, batch):
        hm_h, hm_w = self.pre_sample.tensors.shape[2], self.pre_sample.tensors.shape[3]
        pre_hm = np.zeros((1, hm_h, hm_w), dtype=np.float32)
        trans = batch['trans_input'][0].cpu().numpy()

        # draw pre_hm with self.pos # pre track
        for bbox in self.get_pos().cpu().numpy():
            bbox[:2] = affine_transform(bbox[:2], trans)
            bbox[2:] = affine_transform(bbox[2:], trans)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, hm_w - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, hm_h - 1)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            # draw gt heatmap with
            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                ct = np.array(
                    [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                draw_umich_gaussian(pre_hm[0], ct_int, radius, k=1)

        pre_hm = torch.from_numpy(pre_hm).cuda().unsqueeze_(0)
        outputs = self.obj_detect(samples=self.sample, pre_samples=self.pre_sample,
                                  features=self.img_features, pos=self.encoder_pos_encoding,
                                  pre_features=self.pre_img_features, pre_pos=self.pre_encoder_pos_encoding,
                                  pre_hm=pre_hm)

        # results filtered by th = pre_th
        results = self.postprocessor(outputs, batch['orig_size'], filter_score=True)[0]
        out_scores, labels_out, out_boxes, pre_cts = results['scores'], results['labels'], results['boxes'], results[
            'pre_cts']

        # filter out non-person class
        filtered_idx = labels_out == 1

        return out_boxes[filtered_idx, :].cuda(), out_scores[filtered_idx].cuda(), pre_cts[filtered_idx].cuda()

    def get_pos(self):
        """Get the positions of all active tracks."""
        if len(self.tracks) == 1:
            pos = self.tracks[0].pos.cuda()
        elif len(self.tracks) > 1:
            pos = torch.cat([t.pos for t in self.tracks], dim=0).cuda()
        else:
            pos = torch.zeros(0, 4).cuda()
        return pos

    def get_features(self):
        """Get the features of all active tracks."""
        if len(self.tracks) == 1:
            features = self.tracks[0].features
        elif len(self.tracks) > 1:
            features = torch.cat([t.features for t in self.tracks], 0)
        else:
            features = torch.zeros(0).cuda()
        return features

    def get_inactive_features(self):
        """Get the features of all inactive tracks."""
        if len(self.inactive_tracks) == 1:
            features = self.inactive_tracks[0].features
        elif len(self.inactive_tracks) > 1:
            features = torch.cat([t.features for t in self.inactive_tracks], 0)
        else:
            features = torch.zeros(0).cuda()
        return features

    def reid(self, blob, new_det_pos, new_det_scores):
        """Tries to ReID inactive tracks with provided detections."""
        new_det_features = [torch.zeros(0).cuda() for _ in range(len(new_det_pos))]

        if self.do_reid:
            new_det_features = self.reid_network.test_rois(
                blob['img'], new_det_pos).detach()

            if len(self.inactive_tracks) >= 1:
                # calculate appearance distances
                dist_mat, pos = [], []
                for t in self.inactive_tracks:
                    dist_mat.append(torch.cat([t.test_features(feat.view(1, -1))
                                               for feat in new_det_features], dim=1))
                    pos.append(t.pos)
                if len(dist_mat) > 1:
                    dist_mat = torch.cat(dist_mat, 0)
                    pos = torch.cat(pos, 0)
                else:
                    dist_mat = dist_mat[0]
                    pos = pos[0]

                # calculate IoU distances
                iou = bbox_overlaps(pos, new_det_pos)
                iou_mask = torch.ge(iou, self.reid_iou_threshold)
                iou_neg_mask = ~iou_mask
                # make all impossible assignments to the same add big value
                dist_mat = dist_mat * iou_mask.float() + iou_neg_mask.float() * 1000
                dist_mat = dist_mat.cpu().numpy()

                row_ind, col_ind = linear_sum_assignment(dist_mat)

                assigned = []
                remove_inactive = []
                for r, c in zip(row_ind, col_ind):
                    if dist_mat[r, c] <= self.reid_sim_threshold:
                        t = self.inactive_tracks[r]
                        self.tracks.append(t)
                        t.count_inactive = 0
                        t.pos = new_det_pos[c].view(1, -1)
                        t.reset_last_pos()
                        t.add_features(new_det_features[c].view(1, -1))
                        assigned.append(c)
                        remove_inactive.append(t)

                for t in remove_inactive:
                    self.inactive_tracks.remove(t)

                keep = torch.Tensor([i for i in range(new_det_pos.size(0)) if i not in assigned]).long().cuda()
                if keep.nelement() > 0:
                    new_det_pos = new_det_pos[keep]
                    new_det_scores = new_det_scores[keep]
                    new_det_features = new_det_features[keep]
                else:
                    new_det_pos = torch.zeros(0).cuda()
                    new_det_scores = torch.zeros(0).cuda()
                    new_det_features = torch.zeros(0).cuda()

        return new_det_pos, new_det_scores, new_det_features

    def get_appearances(self, blob):
        """Uses the siamese CNN to get the features for all active tracks."""
        new_features = self.reid_network.test_rois(blob['img'], self.get_pos()).detach()
        return new_features

    def add_features(self, new_features):
        """Adds new appearance features to active tracks."""
        for t, f in zip(self.tracks, new_features):
            t.add_features(f.view(1, -1))

    def align(self, blob):
        """Aligns the positions of active and inactive tracks depending on camera motion."""

        if self.im_index > 0:

            if self.do_reid:
                for t in self.inactive_tracks:
                    # todo check shape and format
                    t.pos = warp_pos(t.pos, self.flow)


    @torch.no_grad()
    def step_reidV3_pre_tracking(self, blob):
        """This function should be called every timestep to perform tracking with a blob
        containing the image information.
        """
        # get backbone features

        # Nested tensor #
        self.sample = NestedTensor(blob['image'].cuda(), blob['pad_mask'].cuda())
        self.img_features, self.encoder_pos_encoding = self.obj_detect.backbone(self.sample)

        if self.pre_img_features is None:
            self.pre_sample = NestedTensor(blob['pre_image'].cuda(), blob['pre_pad_mask'].cuda())
            self.pre_img_features, self.pre_encoder_pos_encoding = self.obj_detect.backbone(self.pre_sample)
        # get flow
        blob['img'] = blob['orig_img'].clone().float() / 255.0
        if self.last_image is not None:
            pre_image = self.last_image
            self.flow = get_flow(pre_image.cuda().float() / 255.0, blob['orig_img'].cuda().float() / 255.0,
                                 pwc_net=self.LiteFlowNet)

        ###########################
        # Look for new detections #
        ###########################
        # detect
        raw_private_det_pos, raw_private_det_scores, pre_cts = self.detect_tracking(blob)
        det_pos = raw_private_det_pos
        det_scores = raw_private_det_scores
        det_pre_cts = pre_cts


        ##################
        # Predict tracks #
        ##################
        if len(self.tracks):
            # align
            if self.do_align:
                # warped inactive tracks and get the pre2curr flow for regression
                self.align(blob)

            person_scores, person_boxes, same_person_scores, [pos_birth,
                                                              scores_birth] = self.tracks_dets_matching_tracking(
                raw_dets=det_pos, raw_scores=det_scores, det_pre_cts=det_pre_cts)

            det_pos, det_scores = pos_birth, scores_birth

            # tracks nms #
            if len(self.tracks):
                # create nms input

                # nms here if tracks overlap # todo do I sill need nms?
                keep = soft_nms_pytorch(self.get_pos(), person_scores, cuda=1)
                self.tracks_to_inactive([self.tracks[i] for i in list(range(len(self.tracks))) if i not in keep])
                #

                if keep.nelement() > 0 and self.do_reid:
                    new_features = self.get_appearances(blob)
                    self.add_features(new_features)

        #####################
        # Create new tracks #
        #####################
        if self.public_detections:
            # no pub dets => in def detect = no private detection
            # case 1: No pub det, private dets OR
            # case 2: No pub det, no private dets

            if len(blob['dets'])==0:
                det_pos = torch.zeros(0, 4).cuda()
                det_scores = torch.zeros(0).cuda()

            # case 3: Pub det, private dets
            elif det_pos.shape[0] > 0:
                curr_dets = torch.cat(blob['dets'], dim=0).cuda()
                # print("curr_dets.shape",curr_dets.shape)
                # using centers
                M = curr_dets.shape[0]
                N = det_pos.shape[0]

                # # iou of shape [#private, #public]#
                iou = bbox_overlaps(det_pos, curr_dets).cpu().numpy()
                # having overlap ?
                valid_det_idx = []
                for j in range(M):
                    # print("pub dets")
                    i = iou[:, j].argmax()
                    if iou[i, j] > 0:
                        iou[i, :] = -1
                        valid_det_idx.append(i)
                # print(valid_det_idx)
                det_pos = det_pos[valid_det_idx]
                det_scores = det_scores[valid_det_idx]
                # print()

            # case 4: No pub det, no private dets
            else:
                det_pos = torch.zeros(0, 4).cuda()
                det_scores = torch.zeros(0).cuda()
        else:
            pass

        if det_pos.nelement() > 0:
            new_det_pos = det_pos
            new_det_scores = det_scores

            # try to re-identify tracks
            new_det_pos, new_det_scores, new_det_features = self.reid(blob, new_det_pos, new_det_scores)

            assert new_det_pos.shape[0] == new_det_features.shape[0] == new_det_scores.shape[0]
            # add new
            if new_det_pos.nelement() > 0:
                self.add(new_det_pos, new_det_scores, new_det_features)

        ####################
        # Generate Results #
        ####################

        # # print()
        for t in self.tracks:
            # print(t)
            if t.id not in self.results.keys():
                self.results[t.id] = {}
            self.results[t.id][self.im_index] = np.concatenate([t.pos[0].cpu().numpy(), np.array([t.score])])

        for t in self.inactive_tracks:
            t.count_inactive += 1

        self.inactive_tracks = [
            t for t in self.inactive_tracks if t.has_positive_area() and t.count_inactive <= self.inactive_patience
        ]

        self.im_index += 1
        self.last_image = blob['orig_img'].clone().float()

        # reuse pre_features
        self.pre_img_features = self.img_features
        self.pre_encoder_pos_encoding = self.encoder_pos_encoding


    @torch.no_grad()
    def step_reidV3_pre_tracking_mot20(self, blob):
        """This function should be called every timestep to perform tracking with a blob
        containing the image information.
        """
        # get backbone features

        # Nested tensor #
        self.sample = NestedTensor(blob['image'].cuda(), blob['pad_mask'].cuda())
        self.img_features, self.encoder_pos_encoding = self.obj_detect.backbone(self.sample)

        if self.pre_img_features is None:
            self.pre_sample = NestedTensor(blob['pre_image'].cuda(), blob['pre_pad_mask'].cuda())
            self.pre_img_features, self.pre_encoder_pos_encoding = self.obj_detect.backbone(self.pre_sample)
        # get flow
        blob['img'] = blob['orig_img'].clone().float() / 255.0
        if self.last_image is not None:
            pre_image = self.last_image
            self.flow = get_flow(pre_image.cuda().float() / 255.0, blob['orig_img'].cuda().float() / 255.0,
                                 pwc_net=self.LiteFlowNet)

        ###########################
        # Look for new detections #
        ###########################
        # detect
        raw_private_det_pos, raw_private_det_scores, pre_cts = self.detect_tracking(blob)
        det_pos = raw_private_det_pos
        det_scores = raw_private_det_scores
        det_pre_cts = pre_cts

        ##################
        # Predict tracks #
        ##################
        if len(self.tracks):
            # align
            if self.do_align:
                # warped inactive tracks and get the pre2curr flow for regression
                self.align(blob)

            person_scores, person_boxes, same_person_scores, [pos_birth,
                                                              scores_birth] = self.tracks_dets_matching_tracking(
                raw_dets=det_pos, raw_scores=det_scores, det_pre_cts=det_pre_cts)

            det_pos, det_scores = pos_birth, scores_birth

            # tracks nms #
            if len(self.tracks):

                keep = soft_nms_pytorch(self.get_pos(), person_scores, cuda=1)
                self.tracks_to_inactive([self.tracks[i] for i in list(range(len(self.tracks))) if i not in keep])
                #

                if keep.nelement() > 0 and self.do_reid:
                    new_features = self.get_appearances(blob)
                    self.add_features(new_features)

        #####################
        # Create new tracks #
        #####################
        if self.public_detections:
            # no pub dets => in def detect = no private detection
            # case 1: No pub det, private dets OR
            # case 2: No pub det, no private dets
            if len(blob['dets']) == 0:
                det_pos = torch.zeros(0, 4).cuda()
                det_scores = torch.zeros(0).cuda()

            # case 3: Pub det, private dets
            elif det_pos.shape[0] > 0:
                curr_dets = torch.cat(blob['dets'], dim=0).cuda()
                # using centers
                M = curr_dets.shape[0]
                N = det_pos.shape[0]

                # # iou of shape [#private, #public]#
                iou = bbox_overlaps(det_pos, curr_dets).cpu().numpy()
                # having overlap ?
                valid_det_idx = []
                for j in range(M):
                    i = iou[:, j].argmax()
                    if iou[i, j] > 0:
                        # print("pub dets")
                        iou[i, :] = -1
                        valid_det_idx.append(i)
                # print(valid_det_idx)
                det_pos = det_pos[valid_det_idx]
                det_scores = det_scores[valid_det_idx]
                # print()

            # case 4: No pub det, no private dets
            else:
                det_pos = torch.zeros(0, 4).cuda()
                det_scores = torch.zeros(0).cuda()
        else:
            pass

        if det_pos.nelement() > 0:
            new_det_pos = det_pos
            new_det_scores = det_scores

            # try to reidentify tracks
            new_det_pos, new_det_scores, new_det_features = self.reid(blob, new_det_pos, new_det_scores)

            assert new_det_pos.shape[0] == new_det_features.shape[0] == new_det_scores.shape[0]
            # add new
            if new_det_pos.nelement() > 0:
                self.add(new_det_pos, new_det_scores, new_det_features)

        ####################
        # Generate Results #
        ####################

        # print()
        for t in self.tracks:
            # print(t)
            if t.id not in self.results.keys():
                self.results[t.id] = {}
            self.results[t.id][self.im_index] = np.concatenate([t.pos[0].cpu().numpy(), np.array([t.score])])

        for t in self.inactive_tracks:
            t.count_inactive += 1

        self.inactive_tracks = [
            t for t in self.inactive_tracks if t.has_positive_area() and t.count_inactive <= self.inactive_patience
        ]

        self.im_index += 1
        self.last_image = blob['orig_img'].clone().float()

        # reuse pre_features
        self.pre_img_features = self.img_features
        self.pre_encoder_pos_encoding = self.encoder_pos_encoding

    def get_results(self):
        return self.results


class Track(object):
    """This class contains all necessary for every individual track."""

    def __init__(self, pos, score, track_id, features, inactive_patience, max_features_num, mm_steps):
        self.id = track_id
        self.pos = pos
        self.score = score
        self.features = deque([features])
        self.ims = deque([])
        self.count_inactive = 0
        self.inactive_patience = inactive_patience
        self.max_features_num = max_features_num
        self.last_pos = deque([pos.clone()], maxlen=mm_steps + 1)
        self.last_v = torch.Tensor([])
        self.gt_id = None

    def has_positive_area(self):
        return self.pos[0, 2] > self.pos[0, 0] and self.pos[0, 3] > self.pos[0, 1]

    def add_features(self, features):
        """Adds new appearance features to the object."""
        self.features.append(features)
        # print(self.max_features_num)
        # print(self.features)
        if len(self.features) > self.max_features_num:
            self.features.popleft()

    def test_features(self, test_features):
        """Compares test_features to features of this Track object"""
        if len(self.features) > 1:
            features = torch.cat(list(self.features), dim=0)
        else:
            features = self.features[0]
        features = features.mean(0, keepdim=True)
        dist = F.pairwise_distance(features, test_features, keepdim=True)
        return dist

    def reset_last_pos(self):
        self.last_pos.clear()
        self.last_pos.append(self.pos.clone())
