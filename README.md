# TE-GCN

Code for the paper ["Temporal-Enhanced Graph Convolution Network for Skeleton-based Action Recognition"](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/cvi2.12086)


Please cite the following paper if you use this repository in your reseach.
```
@article{xie_tegcn_2022,
author = {Xie, Yulai and Zhang, Yang and Ren, Fang},
doi = {https://doi.org/10.1049/cvi2.12086},
journal = {IET Comput.Vis.},
title = {{Temporal-enhanced graph convolution network for skeleton-based action recognition}},
year = {2022}
}
```

Note that:
- This code is based on [2s-AGCN](https://github.com/lshiwjx/2s-AGCN)

## Data preparation
Prepare the data according to [UAVHuman-Pose processing](https://github.com/xieyulai/UAVHuman_For_TE-GCN)

Your `data/` should be like this:
```
uav
___ xsub1
    ___ test_data.npy
    ___ test_label.pkl
    ___ train_data.npy
    ___ train_label.pkl
___ xsub2
    ___ test_data.npy
    ___ test_label.pkl
    ___ train_data.npy
    ___ train_label.pkl

```

## TRAIN
You can train the your model using the scripts:
```
sh scripts/TRAIN_V1.sh
sh scripts/TRAIN_V2.sh
```

## TEST
You can test the your model using the scripts:
```
sh scripts/EVAL_V1.sh
sh scripts/EVAL_V2.sh
```

## WEIGHTS
We have released two trained weights in [baidupan](https://pan.baidu.com/s/1kourPFzEChrjc8kPO0y6rw),passwd is `nwhu`

Your should put them into `runs/`.

- V1:TOP1-42.37%
- V2:TOP1-68.11%
