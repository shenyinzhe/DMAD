# nuScenes
> Modified from UniAD.


Download nuScenes V1.0 full dataset data, CAN bus and map(v1.3) extensions [HERE](https://www.nuscenes.org/download), then follow the steps below to prepare the data.


**Download nuScenes, CAN_bus and Map extensions**
```shell
cd DMAD
mkdir data
# Download nuScenes V1.0 full dataset data directly to (or soft link to) DMAD/data/
# Download CAN_bus and Map(v1.3) extensions directly to (or soft link to) DMAD/data/nuscenes/
```

**Prepare data info**


*Option1: We have already prepared the off-the-shelf data infos for you:*
```shell
cd DMAD/data
mkdir infos && cd infos
wget https://github.com/shenyinzhe/DMAD/releases/download/v1.0/nuscenes_mini_infos_temporal_train.pkl  # train_infos
wget https://github.com/shenyinzhe/DMAD/releases/download/v1.0/nuscenes_mini_infos_temporal_val.pkl  # val_infos

# mini set:
wget https://github.com/shenyinzhe/DMAD/releases/download/v1.0/nuscenes_mini_infos_temporal_train.pkl
wget https://github.com/shenyinzhe/DMAD/releases/download/v1.0/nuscenes_mini_infos_temporal_val.pkl
```


*Option2: You can also generate the data infos by yourself:*
```shell
cd DMAD/data
mkdir infos
./tools/uniad_create_data.sh
# This will generate nuscenes_infos_temporal_{train,val}.pkl
```

**Prepare Motion Anchors**
```shell
cd DMAD/data
mkdir others && cd others
wget https://github.com/shenyinzhe/DMAD/releases/download/v1.0/motion_anchor_infos_mode6.pkl
```

**The Overall Structure**

*Please make sure the structure of DMAD is as follows:*
```
DMAD
├── projects/
├── tools/
├── ckpts/
│   ├── bevformer_r101_dcn_24ep.pth
│   ├── dmad_stage1.pth
|   ├── dmad_stage2.pth
├── data/
│   ├── nuscenes/
│   │   ├── can_bus/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── v1.0-test/
│   │   ├── v1.0-trainval/
│   ├── infos/
│   │   ├── nuscenes_infos_temporal_train.pkl
│   │   ├── nuscenes_infos_temporal_val.pkl
│   ├── others/
│   │   ├── motion_anchor_infos_mode6.pkl
```
---
<- Last Page:  [Installation](./INSTALL.md)

-> Next Page: [Train/Eval](./TRAIN_EVAL.md)