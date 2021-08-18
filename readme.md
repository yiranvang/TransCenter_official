## TransCenter: Transformers with Dense Queries for Multiple-Object Tracking <br />

**TransCenter: Transformers with Dense Queries for Multiple-Object Tracking** <br />
[Yihong Xu](https://team.inria.fr/perception/team-members/yihong-xu/), [Yutong Ban](https://team.inria.fr/perception/team-members/yutong-ban/), [Guillaume Delorme](https://team.inria.fr/robotlearn/team-members/guillaume-delorme/), [Chuang Gan](https://people.csail.mit.edu/ganchuang/), [Daniela Rus](http://danielarus.csail.mit.edu/), [Xavier Alameda-Pineda](http://xavirema.eu/) <br />
**[[Paper](https://arxiv.org/abs/2103.15145)]** <br />

<div align="center">
  <img src="https://github.com/yihongXU/TransCenter/raw/main/pipeline.png" width="1200px" />
</div>

## Bibtex
If you find this code useful, please star the project and consider citing:

```
@misc{xu2021transcenter,
      title={TransCenter: Transformers with Dense Queries for Multiple-Object Tracking}, 
      author={Yihong Xu and Yutong Ban and Guillaume Delorme and Chuang Gan and Daniela Rus and Xavier Alameda-Pineda},
      year={2021},
      eprint={2103.15145},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Environment Preparation 
### Option 1 (recommended):
We provide two singularity images (similar to docker) containing all the packages we need for TransCenter:

1) Install singularity > 3.7.1:  [https://sylabs.io/guides/3.0/user-guide/installation.html#install-on-linux](https://sylabs.io/guides/3.0/user-guide/installation.html#install-on-linux)
2) Download one of the singularity images:

[**pytorch1-5cuda10-1.sif**](https://drive.google.com/file/d/1MDNwMzJnculxEvEs3KN_rDE6XoYojx1H/view?usp=sharing) tested with Nvidia GTX TITAN. Or

[**pytorch1-5cuda10-1_RTX.sif**](https://drive.google.com/file/d/1s4rgDv05zq7nPlsD5EQyK5sYAcE9yFeA/view?usp=sharing) tested with Nvidia RTX TITAN, Quadro RTX 8000, RTX 2080Ti, Quadro RTX 4000.

- Launch a Singularity image
```shell
singularity shell --nv --bind yourLocalPath:yourPathInsideImage YourSingularityImage.sif
```
**- -bind: to link a singularity path with a local path. By doing this, you can find data from local PC inside Singularity image;** <br />
**- -nv: use the local Nvidia driver.**
### Option 2:

You can also build your own environment:
1) we use anaconda to simplify the package installations, you can download anaconda (4.9.2) here: [https://www.anaconda.com/products/individual](https://www.anaconda.com/products/individual)
2) you can create your conda env by doing 
```
conda create --name <YourEnvName> --file requirements.txt
```
3) TransCenter uses Deformable transformer from Deformable DETR. Therefore, we need to install deformable attention modules:
```
cd ./to_install/ops
sh ./make.sh
# unit test (should see all checking is True)
python test.py
```
4) TransCenter uses pytorch-liteflownet during tracking, which depends on *correlation_package*. You can install it by doing:
```
cd ./to_install/correlation_package
python setup.py install
```
5) for the up-scale and merge module in TransCenter, we use deformable convolution module, you can install it with:
```
cd ./to_install/DCNv2
./make.sh         # build
python testcpu.py    # run examples and gradient check on cpu
python testcuda.py   # run examples and gradient check on gpu
```
see also known issues from [https://github.com/CharlesShang/DCNv2](https://github.com/CharlesShang/DCNv2).
If you have issues related to cuda of the third-party modules, please try to recompile them in the GPU that you use for training and testing. 
The dependencies are compatible with Pytorch 1.5, cuda 10.2.

## Data Preparation ##
[ms coco](https://cocodataset.org/#download): we use only the *person* category for pretraining TransCenter. The code for filtering is provided in ./data/coco_person.py.
```
@inproceedings{lin2014microsoft,
  title={Microsoft coco: Common objects in context},
  author={Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and Hays, James and Perona, Pietro and Ramanan, Deva and Doll{\'a}r, Piotr and Zitnick, C Lawrence},
  booktitle={European conference on computer vision},
  pages={740--755},
  year={2014},
  organization={Springer}
}
```

[CrowdHuman](https://www.crowdhuman.org/): CrowdHuman labels are converted to coco format, the conversion can be done through ./data/convert_crowdhuman_to_coco.py.

```
@article{shao2018crowdhuman,
    title={CrowdHuman: A Benchmark for Detecting Human in a Crowd},
    author={Shao, Shuai and Zhao, Zijian and Li, Boxun and Xiao, Tete and Yu, Gang and Zhang, Xiangyu and Sun, Jian},
    journal={arXiv preprint arXiv:1805.00123},
    year={2018}
  }
```

[MOT17](https://motchallenge.net/data/MOT17/): MOT17 labels are converted to coco format, the conversion can be done through ./data/convert_mot_to_coco.py.

```
@article{milan2016mot16,
  title={MOT16: A benchmark for multi-object tracking},
  author={Milan, Anton and Leal-Taix{\'e}, Laura and Reid, Ian and Roth, Stefan and Schindler, Konrad},
  journal={arXiv preprint arXiv:1603.00831},
  year={2016}
}
```

[MOT20](https://motchallenge.net/data/MOT20/): MOT20 labels are converted to coco format, the conversion can be done through ./data/convert_mot20_to_coco.py.

```
@article{dendorfer2020mot20,
  title={Mot20: A benchmark for multi object tracking in crowded scenes},
  author={Dendorfer, Patrick and Rezatofighi, Hamid and Milan, Anton and Shi, Javen and Cremers, Daniel and Reid, Ian and Roth, Stefan and Schindler, Konrad and Leal-Taix{\'e}, Laura},
  journal={arXiv preprint arXiv:2003.09003},
  year={2020}
}
```

We also provide the filtered/converted labels:

[ms coco person labels](https://drive.google.com/drive/folders/1PuVXRQV10fRW8MTBG8txhamSJqaSWBbc?usp=sharing): please put the *annotations* folder inside *cocoperson* to your ms coco dataset root folder.

[CrowdHuman coco format labels](https://drive.google.com/drive/folders/152K_-FjltstDPkW3jKUEaRHrtxes6mr8?usp=sharing): please put the *annotations* folder inside *crowdhuman* to your CrowdHuman dataset root folder.

[MOT17 coco format labels](https://drive.google.com/drive/folders/1SxaVF4KddLp7t_twF53wpOifrLNzDNXE?usp=sharing): please put the *annotations* and *annotations_onlySDP* folders inside *MOT17* to your MOT17 dataset root folder.

[MOT20 coco format labels](https://drive.google.com/drive/folders/12svjv5V7-pC2BHJfyxfo_9wYcEUWGs27?usp=sharing): please put the *annotations* folder inside *MOT20* to your MOT20 dataset root folder.


## Model Zoo
[deformable transformer pretrained](https://drive.google.com/file/d/1UOUa8m_FRPhPUGr4iDHIcVC7wt7-A0gP/view?usp=sharing): pretrained model from deformable-DETR.

[coco_pretrained](https://drive.google.com/file/d/1BAQ7Xw5dP8oHcTqBwpzsE3GtyRYrKfzd/view?usp=sharing): model trained with coco person dataset.

[CH_pretrained](https://drive.google.com/file/d/1rjbHIy3txFd4nKttfrhpIoxlMsKFTnFI/view?usp=sharing): model pretrained on coco person and fine-tuned on CrowdHuman dataset.

[MOT17_fromCoCo](https://drive.google.com/file/d/1UlklwB64CdI9sYrC16FUpe3zI1Vp4xEK/view?usp=sharing): model pretrained on coco person and fine-tuned on MOT17 trainset.

[MOT17_fromCH](https://drive.google.com/file/d/1mpcDJG6eHVLRulHrXIYxYMT9dn5y-BES/view?usp=sharing): model pretrained on CrowdHuman and fine-tuned on MOT17 trainset.

[MOT20_fromCoCo](https://drive.google.com/file/d/146oKnZY77cj9cCzY_hQ0rAC86tkIrU0J/view?usp=sharing): model pretrained on coco person and fine-tuned on MOT20 trainset.

[MOT20_fromCH](https://drive.google.com/file/d/1an6oGLJKVLvvcenZOhT2mwLsm4Q5LCZh/view?usp=sharing): model pretrained on CrowdHuman and fine-tuned on MOT20 trainset.

Please put all the pretrained models to *./model_zoo* .
## Training

- Pretrained on coco person dataset:
```
cd TransCenter_official
python -m torch.distributed.launch --nproc_per_node=4 --use_env ./training/transcenter/main_coco_tracking.py --output_dir=./output/whole_coco --batch_size=4 --num_workers=20 --resume=./model_zoo/r50_deformable_detr-checkpoint.pth --pre_hm --tracking --data_dir=/scratch2/scorpio/yixu/cocodataset/ --epochs=11
```
- Pretrained on CrowdHuman dataset:
```
cd TransCenter_official
python -m torch.distributed.launch --nproc_per_node=4 --use_env ./training/transcenter/main_crowdHuman_tracking.py --output_dir=./output/whole_ch_from_COCO --batch_size=4 --num_workers=20 --resume=./model_zoo/coco_pretrained.pth --pre_hm --tracking --data_dir=/scratch2/scorpio/yixu/crowd_human/ --epochs=49
```

- Train MOT17 from CoCo pretrained model:
```
cd TransCenter_official
python -m torch.distributed.launch --nproc_per_node=2 --use_env ./training/transcenter/main_mot17_tracking.py --output_dir=./output/whole_MOT17_from_COCO --batch_size=2 --num_workers=20 --resume=./model_zoo/coco_pretrained.pth --pre_hm --tracking  --same_aug_pre --image_blur_aug --data_dir=/scratch/scorpio/yixu/rawdata/MOT17/ --epochs=33
```

- Train MOT17 from CrowdHuman pretrained model:
```
cd TransCenter_official
python -m torch.distributed.launch --nproc_per_node=2 --use_env ./training/transcenter/main_mot17_tracking.py --output_dir=./output/whole_MOT17_from_CH --batch_size=2 --num_workers=20 --resume=./model_zoo/CH_pretrained.pth --pre_hm --tracking  --same_aug_pre --image_blur_aug --data_dir=/scratch/scorpio/yixu/rawdata/MOT17/ --epochs=23
```

- Train MOT20 from CoCo pretrained model:
```
cd TransCenter_official
python -m torch.distributed.launch --nproc_per_node=2 --use_env ./training/transcenter/main_mot20_tracking.py --output_dir=./output/whole_MOT20_from_COCO --batch_size=2 --num_workers=20 --resume=./model_zoo/coco_pretrained.pth --pre_hm --tracking  --same_aug_pre --image_blur_aug --not_max_crop --data_dir=/scratch/scorpio/yixu/rawdata/MOT20/ --epochs=19
```

- Train MOT20 from CrowdHuman pretrained model:
```
cd TransCenter_official
python -m torch.distributed.launch --nproc_per_node=2 --use_env ./training/transcenter/main_mot20_tracking.py --output_dir=./output/whole_MOT20_from_CH --batch_size=2 --num_workers=20 --resume=./model_zoo/CH_pretrained.pth --pre_hm --tracking  --same_aug_pre --image_blur_aug --not_max_crop --data_dir=/scratch/scorpio/yixu/rawdata/MOT20/ --epochs=39
```

Tips:
1) If you encounter *RuntimeError: cuDNN error: CUDNN_STATUS_INTERNAL_ERROR* in some GPUs, please try to set *torch.backends.cudnn.benchmark=False*.
2) The number of epochs is by default 50, the listed numbers of epochs are where models having best eval scores (det mAP).
3) Depending on your environment and GPUs, you might experience MOTA jitter in your final models.
4) You may see training noise during fine-tuning, especially for MOT17/MOT20 training with well-pretrained models. You can slow down the training rate by 1/10, apply early stopping, increase batch size with GPUs having more memory.  
5) If you have GPU memory issues, try to lower the batch size for training and evaluation in main_****.py, freeze the resnet backbone and use our coco/CH pretrained models.


## Tracking
Using Public detections:

- MOT17:
```
cd TransCenter_official
python ./tracking/transcenter/mot17_pub.py --data_dir=/scratch/scorpio/yixu/rawdata/MOT17/
```
- MOT20:
```
cd TransCenter_official
python ./tracking/transcenter/mot20_pub.py --data_dir=/scratch/scorpio/yixu/rawdata/MOT20/
```

Using Private detections:

- MOT17:
```
cd TransCenter_official
python ./tracking/transcenter/mot17_private.py --data_dir=/scratch/scorpio/yixu/rawdata/MOT17/
```
- MOT20:
```
cd TransCenter_official
python ./tracking/transcenter/mot20_private.py --data_dir=/scratch/scorpio/yixu/rawdata/MOT20/
```

Notes:
1) we recently corrected an image loading bug during reading certain images having image ratio close to 1 (in MOT20) in the code, bringing better performance in MOT20.
2) you can test your model by changing the model_path inside *mot17[20]_private[pub].py*.

## MOTChallenge Results
MOT17 public detections:
     
| Pretrained| MOTA     | MOTP     | IDF1 |  FP    | FN    | IDS |
|-----------|----------|----------|--------|-------|------|----------------|
|   CoCo  |  68.8%   |  79.9%   | 61.4% | 22,860  | 149,188  |     4,102     |
|   CH    |  71.9%   |  81.4%   | 62.3%  | 17,378 | 137,008 |     4,046    |

MOT20 public detections:
   
| Pretrained| MOTA     | MOTP     | IDF1 |  FP    | FN    | IDS |
|-----------|----------|----------|--------|-------|------|----------------|
|   CoCo    |  61.0%   |  79.5%   | 49.8%  | 49,189   | 147,890  |     4,493     |
|   CH      |  62.3%   |  79.9%   | 50.3%  | 43,006  | 147,505  |     4,545     |


MOT17 private detections:
   
| Pretrained| MOTA     | MOTP     | IDF1 |  FP    | FN    | IDS |
|-----------|----------|----------|--------|-------|------|----------------|
|   CoCo  |  70.0%   |  79.6%   | 62.1% | 28,119   | 136,722  |    4,647     |
|   CH    |  73.2%   |  81.1%   | 62.2% | 23,112 | 123,738 |     4,614    |

MOT20 private detections:

| Pretrained| MOTA     | MOTP     | IDF1 |  FP    | FN    | IDS |
|-----------|----------|----------|--------|-------|------|----------------|
|   CoCo   |  60.6%   |  79.5%   | 49.6% | 52,332  | 146,809 |     4,604     |
|   CH   |  61.9%   |  79.9%   | 50.4%  | 45,895  | 146,347  |     4,653     |


**Note:** 
- The results can be slightly different depending on the running environment.
- Knowing the work is under review, we might keep updating the results in the near future.

## Acknowledgement

The code for TransCenter is modified and network pre-trained weights are obtained from the following repositories:

1) The Person Reid Network  (./tracking/transcenter/model_zoo/ResNet_iter_25245.pth) is from Tracktor.
2) The lightflownet pretrained model (./tracking/transcenter/util/LiteFlownet/network-kitti.pytorch) is from pytorch-liteflownet and LiteFlowNet.
3) The deformable transformer pretrained model (./model_zoo/r50_deformable_detr-checkpoint.pth) is from Deformable-DETR.
4) The data format conversion code is modified from CenterTrack.

[**CenterTrack**](https://github.com/xingyizhou/CenterTrack), [**Deformable-DETR**](https://github.com/fundamentalvision/Deformable-DETR), [**Tracktor**](https://github.com/phil-bergmann/tracking_wo_bnw).
```
@article{zhou2020tracking,
  title={Tracking Objects as Points},
  author={Zhou, Xingyi and Koltun, Vladlen and Kr{\"a}henb{\"u}hl, Philipp},
  journal={ECCV},
  year={2020}
}

@InProceedings{tracktor_2019_ICCV,
author = {Bergmann, Philipp and Meinhardt, Tim and Leal{-}Taix{\'{e}}, Laura},
title = {Tracking Without Bells and Whistles},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
month = {October},
year = {2019}}

@article{zhu2020deformable,
  title={Deformable DETR: Deformable Transformers for End-to-End Object Detection},
  author={Zhu, Xizhou and Su, Weijie and Lu, Lewei and Li, Bin and Wang, Xiaogang and Dai, Jifeng},
  journal={arXiv preprint arXiv:2010.04159},
  year={2020}
}
```
Several modules are from:

**MOT Metrics in Python**: [**py-motmetrics**](https://github.com/cheind/py-motmetrics)

**Soft-NMS**: [**Soft-NMS**](https://github.com/DocF/Soft-NMS)

**DETR**: [**DETR**](https://github.com/facebookresearch/detr)

**DCNv2**: [**DCNv2**](https://github.com/CharlesShang/DCNv2)

**correlation_package**: [**correlation_package**](https://github.com/NVIDIA/flownet2-pytorch/tree/master/networks/correlation_package)

**pytorch-liteflownet**: [**pytorch-liteflownet**](https://github.com/sniklaus/pytorch-liteflownet)

**LiteFlowNet**: [**LiteFlowNet**](https://github.com/twhui/LiteFlowNet)
```
@InProceedings{hui18liteflownet,
    author = {Tak-Wai Hui and Xiaoou Tang and Chen Change Loy},
    title = {LiteFlowNet: A Lightweight Convolutional Neural Network for Optical Flow Estimation},
    booktitle  = {Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year = {2018},
    pages = {8981--8989},
    }
```

