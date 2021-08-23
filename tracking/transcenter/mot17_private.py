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
import os
import torch

import numpy as np
from datasets.transcenter_dataset.mot17_val_save_mem import MOT17_val
from reid.resnet import resnet50
from util.LiteFlownet.light_flownet import Network
import csv
import os.path as osp
import yaml
from tracker import Tracker
from models import build_model
import argparse
from torch.utils.data import DataLoader

from shutil import copyfile
from util.image import get_affine_transform

torch.set_grad_enabled(False)

curr_pth = '/'.join(osp.dirname(__file__).split('/'))

def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=70, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--sgd', action='store_true')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')
    parser.add_argument('--learnable_queries', action='store_true',
                        help="If true, we use learnable parameters.")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    parser.add_argument('--heads', default=['hm', 'reg', 'wh', 'center_offset', 'tracking'], type=str, nargs='+')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Loss coefficients
    parser.add_argument('--hm_weight', default=1, type=float)
    parser.add_argument('--off_weight', default=1, type=float)
    parser.add_argument('--wh_weight', default=0.1, type=float)
    parser.add_argument('--tracking_weight', default=1, type=float)
    parser.add_argument('--ct_offset_weight', default=0.1, type=float)
    parser.add_argument('--boxes_weight', default=0.5, type=float)
    parser.add_argument('--giou_weight', default=0.4, type=float)
    parser.add_argument('--norm_factor', default=1.0, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='mot17')
    parser.add_argument('--data_dir', default='MOT17', type=str)

    parser.add_argument('--coco_panargsic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')

    # centers
    parser.add_argument('--num_classes', default=1, type=int)
    parser.add_argument('--input_h', default=640, type=int)
    parser.add_argument('--input_w', default=1088, type=int)
    parser.add_argument('--down_ratio', default=4, type=int)
    parser.add_argument('--dense_reg', type=int, default=1, help='')
    parser.add_argument('--trainval', action='store_true',
                             help='include validation in training and '
                                  'test on test set')

    parser.add_argument('--K', type=int, default=300,
                             help='max number of output objects.')

    parser.add_argument('--debug', action='store_true')

    # noise
    parser.add_argument('--not_rand_crop', action='store_true',
                             help='not use the random crop data augmentation'
                                  'from CornerNet.')
    parser.add_argument('--not_max_crop', action='store_true',
                             help='used when the training dataset has'
                                  'inbalanced aspect ratios.')
    parser.add_argument('--shift', type=float, default=0.0,
                             help='when not using random crop'
                                  'apply shift augmentation.')
    parser.add_argument('--scale', type=float, default=0.0,
                             help='when not using random crop'
                                  'apply scale augmentation.')
    parser.add_argument('--rotate', type=float, default=0,
                             help='when not using random crop'
                                  'apply rotation augmentation.')
    parser.add_argument('--flip', type = float, default=0.0,
                             help='probability of applying flip augmentation.')
    parser.add_argument('--no_color_aug', action='store_true',
                             help='not use the color augmenation '
                                  'from CornerNet')
    parser.add_argument('--aug_rot', type=float, default=0,
                             help='probability of applying '
                                  'rotation augmentation.')

    # tracking
    parser.add_argument('--max_frame_dist', type=int, default=3)
    parser.add_argument('--merge_mode', type=int, default=1)
    parser.add_argument('--tracking', action='store_true')
    parser.add_argument('--pre_hm', action='store_true')
    parser.add_argument('--same_aug_pre', action='store_true')
    parser.add_argument('--zero_pre_hm', action='store_true')
    parser.add_argument('--hm_disturb', type=float, default=0.00)
    parser.add_argument('--lost_disturb', type=float, default=0.0)
    parser.add_argument('--fp_disturb', type=float, default=0.0)
    parser.add_argument('--pre_thresh', type=float, default=-1)
    parser.add_argument('--track_thresh', type=float, default=0.3)
    parser.add_argument('--new_thresh', type=float, default=0.3)
    parser.add_argument('--ltrb_amodal', action='store_true')
    parser.add_argument('--ltrb_amodal_weight', type=float, default=0.1)
    parser.add_argument('--public_det', action='store_true')
    parser.add_argument('--no_pre_img', action='store_true')
    parser.add_argument('--zero_tracking', action='store_true')
    parser.add_argument('--hungarian', action='store_true')
    parser.add_argument('--max_age', type=int, default=-1)
    parser.add_argument('--out_thresh', type=float, default=-1,
                             help='')
    return parser


def write_results(all_tracks, out_dir, seq_name=None, frame_offset=0):
    output_dir = out_dir + "/txt/"
    """Write the tracks in the format for MOT16/MOT17 submission

    all_tracks: dictionary with 1 dictionary for every track with {..., i:np.array([x1,y1,x2,y2]), ...} at key track_num

    Each file contains these lines:
    <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
    """
    #format_str = "{}, -1, {}, {}, {}, {}, {}, -1, -1, -1"

    assert seq_name is not None, "[!] No seq_name, probably using combined database"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file = osp.join(output_dir, seq_name+'.txt')

    with open(file, "w") as of:
        writer = csv.writer(of, delimiter=',')
        for i, track in all_tracks.items():
            for frame, bb in track.items():
                x1 = bb[0]
                y1 = bb[1]
                x2 = bb[2]
                y2 = bb[3]
                writer.writerow([frame+frame_offset, i+1, x1+1, y1+1, x2-x1+1, y2-y1+1, -1, -1, -1, -1])

    # copy to FRCNN, DPM.txt, private setting
    copyfile(file, file[:-7]+"FRCNN.txt")
    copyfile(file, file[:-7]+"DPM.txt")


def main(tracktor, reid):
    torch.manual_seed(tracktor['seed'])
    torch.cuda.manual_seed(tracktor['seed'])
    np.random.seed(tracktor['seed'])
    torch.backends.cudnn.deterministic = True

    # load model
    main_args = get_args_parser().parse_args()
    main_args.node0 = True
    main_args.private = True
    ds = MOT17_val(main_args, 'test')

    main_args.input_h, main_args.input_w = ds.default_resolution[0], ds.default_resolution[1]
    print(main_args.input_h, main_args.input_w)
    main_args.output_h = main_args.input_h // main_args.down_ratio
    main_args.output_w = main_args.input_w // main_args.down_ratio
    main_args.input_res = max(main_args.input_h, main_args.input_w)
    main_args.output_res = max(main_args.output_h, main_args.output_w)
    # threshold
    main_args.track_thresh = tracktor['tracker']["track_thresh"]
    main_args.pre_thresh = tracktor['tracker']["pre_thresh"]
    main_args.new_thresh = max(tracktor['tracker']["track_thresh"], tracktor['tracker']["new_thresh"])

    model, criterion, postprocessors = build_model(main_args)

    model.cuda()
    model.eval()
    # load flowNet
    liteFlowNet = Network().cuda().eval()

    # load reid network
    reid_network = resnet50(pretrained=False, **reid['cnn'])
    print(f"Loading Reid Model {tracktor['reid_weights']}")
    reid_network.load_state_dict(torch.load(curr_pth + "/model_zoo/" + tracktor['reid_weights'],
                                            map_location=lambda storage, loc: storage))
    reid_network.eval()
    reid_network.cuda()

    # tracker
    tracker = Tracker(model, reid_network, liteFlowNet, tracktor['tracker'], postprocessor=postprocessors['bbox'], main_args=main_args)
    tracker.public_detections = False

    # dataloader
    data_loader = DataLoader(ds, 1, shuffle=False,
                                 drop_last=False, num_workers=0,
                                 pin_memory=True)

    models = [
        # "./model_zoo/MOT17_fromCoCo.pth",
        "./model_zoo/MOT17_fromCH.pth"
    ]
    output_dirs = [
        # curr_pth + '/test_models/mot17_fromCoCo_test_private/',
        curr_pth + '/test_models/mot17_fromCH_test_private/',

    ]

    for model_dir, output_dir in zip(models, output_dirs):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        seq_num = 'SDP'
        tracktor['obj_detect_model'] = model_dir
        tracktor['output_dir'] = output_dir
        print("Loading: ", tracktor['obj_detect_model'])
        model.load_state_dict(torch.load(tracktor['obj_detect_model'])["model"])

        model.cuda()
        model.eval()
        for seq, seq_n in data_loader:
            seq_name = seq_n[0]
            print("seq_name: ", seq_name)

            if seq_num not in seq_name:
                del seq
                continue

            if os.path.exists(output_dir + "txt/" + seq_name + '.txt'):
                print(output_dir + "txt/" + seq_name + '.txt exists.')
                del seq
                continue
            tracker.reset()
            keys = list(seq.keys())
            keys.pop(keys.index('v_id'))
            frames_list = sorted(keys)
            frame_offset = 0

            v_id = seq["v_id"].item()
            pub_dets = ds.VidPubDet[v_id]

            c = None
            s = None
            trans_input = None
            # print(seq)
            for idx, frame_name in enumerate(frames_list):
                # print(batch)
                blob = seq[frame_name]
                # print(blob["frame_id"])
                frame_id = blob["frame_id"].item()

                img_id = blob["img_id"].item()
                # load batch
                # starts with 0 #
                pub_det = pub_dets[frame_id - 1]

                img, anns, img_info, _, pad_mask = ds._load_data(img_id)

                height, width = img.shape[0], img.shape[1]
                if c is None:
                    # get image centers
                    c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
                    # get image size or max h or max w
                    s = max(img.shape[0], img.shape[1]) * 1.0 if not ds.opt.not_max_crop \
                        else np.array([img.shape[1], img.shape[0]], np.float32)

                aug_s, rot, flipped = 1, 0, 0
                if trans_input is None:
                    # we will reshape image to standard input shape,
                    # trans_input =transform for resizing input to input size
                    trans_input = get_affine_transform(c, s, rot, [ds.opt.input_w, ds.opt.input_h])
                # the output heatmap size != input size, trans_output = transform for resizing input to output size
                inp, padding_mask = ds._get_input(img, trans_input, padding_mask=pad_mask)

                # load_pre #
                # select a pre image with random interval
                pre_image, _, frame_dist, pre_img_id, pre_pad_mask = ds._load_pre_data(img_info['video_id'], img_info['frame_id'])
                pre_inp, pre_padding_mask = ds._get_input(pre_image, trans_input, padding_mask=pre_pad_mask)

                #
                batch = {'image': torch.from_numpy(inp).unsqueeze_(0).cuda(),
                         'pad_mask': torch.from_numpy(padding_mask.astype(np.bool)).unsqueeze_(0).cuda(),
                         'pre_image': torch.from_numpy(pre_inp).unsqueeze_(0).cuda(),
                         'pre_pad_mask': torch.from_numpy(pre_padding_mask.astype(np.bool)).unsqueeze_(0).cuda(),
                         'trans_input': torch.from_numpy(trans_input).unsqueeze_(0).cuda(),
                         "frame_dist": frame_dist,
                         'orig_size': torch.from_numpy(np.asarray([height, width])).unsqueeze_(0).cuda(),
                         'dets': [torch.tensor(p_det).float().unsqueeze_(0) for p_det in pub_det],
                         'orig_img': torch.from_numpy(
                             np.ascontiguousarray(img.transpose(2, 0, 1)).astype(np.float32)).unsqueeze_(0)}

                if idx == 0:
                    frame_offset = int(frame_name[:-4])
                    print("frame offset : ", frame_offset)

                print("step frame: ", int(frame_name[:-4]))

                batch['frame_name'] = frame_name
                batch['video_name'] = seq_name

                tracker.step_reidV3_pre_tracking(batch)

                del pre_inp
                del pre_image
                del img
                del inp
                del pre_padding_mask
                del padding_mask
                del batch
                del pub_det
            if seq_num in seq_name and not os.path.exists(output_dir + "txt/" + seq_name + '.txt'):
                # save results #
                results = tracker.get_results()
                write_results(results, tracktor['output_dir'], seq_name=seq_name, frame_offset=frame_offset)

            del seq
            del pub_dets


with open(curr_pth + '/cfgs/detracker_reidV3.yaml', 'r') as f:
    tracktor = yaml.load(f)['tracktor']

with open(curr_pth+ '/cfgs/reid.yaml', 'r') as f:
    reid = yaml.load(f)['reid']
    # print(reid)

main(tracktor, reid)
