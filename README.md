# TE-GCN
Code for the paper "Temporal-Enhanced Graph Convolution Network for Skeleton-based Action Recognition"
(It is stll under construction)
The code is based on [2s-AGCN](https://github.com/lshiwjx/2s-AGCN)

## Data preparation
Preparing your data according to this repo [UAVHuman-Pose](https://github.com/xieyulai/UAVHuman_For_TE-GCN)

Your data/ should be like this,
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
You can train the your model using the scripts,
```
sh scripts/TRAIN_V1.sh
sh scripts/TRAIN_V2.sh
```

## TEST
You can test the your model using the scripts,
```
sh scripts/EVAL_V1.sh
sh scripts/EVAL_V2.sh
```

## WEIGHTS
We release two trained weight in [pan.baidu.com]()
Your should put them into 'runs/'.

V1:TOP1-42.37%
V2:TOP1-68.11%
