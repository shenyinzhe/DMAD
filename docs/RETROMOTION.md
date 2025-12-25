# RetroMotion

- Prepare model weights and anchors
```shell
wget https://github.com/shenyinzhe/DMAD/releases/download/v1.0/dmad_stage2.pth
wget https://github.com/shenyinzhe/DMAD/releases/download/v1.0/motion_anchor_infos_mode6.pkl
wget https://github.com/shenyinzhe/DMAD/releases/download/v1.0/nuscenes_infos_temporal_train.pkl  # train_infos
wget https://github.com/shenyinzhe/DMAD/releases/download/v1.0/nuscenes_infos_temporal_val.pkl  # val_infos
```

- Build apptainer
```shell
docker build -t dmad:latest -f docker/Dockerfile .
singularity build dmad.sif docker-daemon://dmad:latest
```

- Folder structure looks like this:
```
DMAD
├── projects/
├── tools/
├── ckpts/
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

Train:
```
apptainer exec --nv \
    --env PYTHONPATH=/workspace/DMAD \
    --env NCCL_P2P_DISABLE=1 \
    -B /storage_local/shen/DMAD:/workspace/DMAD \
    -B /tmp/shen/nuscenes:/tmp/shen/nuscenes \
    --pwd /workspace/DMAD \
    dmad.sif \
    python -m torch.distributed.launch --nproc_per_node=4 \
    --master_addr="127.0.0.1" --master_port=28593 \
    --nnodes=1 --node_rank=0 \
    tools/train.py projects/configs/stage2_e2e/dmad_stage2.py \
    --launcher pytorch \
    --deterministic \
    --work-dir work_dirs/dmad_stage2
```

Eval:
```
apptainer exec --nv \
    --env PYTHONPATH=/workspace/DMAD \
    --env NCCL_P2P_DISABLE=1 \
    -B /storage_local/shen/DMAD:/workspace/DMAD \
    -B /tmp/shen/nuscenes:/tmp/shen/nuscenes \
    --pwd /workspace/DMAD \
    dmad.sif \
    python -m torch.distributed.launch --nproc_per_node=4 \
    --master_addr="127.0.0.1" --master_port=28691 \
    --nnodes=1 --node_rank=0 \
    tools/test.py projects/configs/stage2_e2e/dmad_stage2.py \
    ckpts/dmad_stage2.pth \
    --launcher pytorch \
    --eval box \
    --show-dir work_dirs/eval \
    --out output/results.pkl
```

## Model outputs

### Map segmentation
- train:
  - lanes, dividers, pedestrian crossings: https://github.com/shenyinzhe/DMAD/blob/main/projects/mmdet3d_plugin/uniad/dense_heads/panseg_head.py#L753
  - drivable area: https://github.com/shenyinzhe/DMAD/blob/main/projects/mmdet3d_plugin/uniad/dense_heads/panseg_head.py#L756
- eval: https://github.com/shenyinzhe/DMAD/blob/main/projects/mmdet3d_plugin/uniad/detectors/uniad_track.py#L893

### Prediction
https://github.com/shenyinzhe/DMAD/blob/main/projects/mmdet3d_plugin/uniad/dense_heads/motion_head.py#L347

- outs['all_traj_preds']: [num_layers, batch_size, num_agents, num_modes, num_steps, waypoint], where the waypoint has 5 entries (x, y, std_x, std_y, correlation)
- outs['all_traj_scores']: [num_layers, batch_size, num_agents, num_modes]
- outs['valid_traj_masks']: [batch_size, num_agents]

### Planning
https://github.com/shenyinzhe/DMAD/blob/main/projects/mmdet3d_plugin/uniad/dense_heads/planning_head.py#L199
- sdc_traj_all: [batch_size, num_steps, waypoint], where the waypoint has 2 entries (x, y)

## Note the planning postprocessing
https://github.com/shenyinzhe/DMAD/blob/main/projects/mmdet3d_plugin/uniad/dense_heads/planning_head.py#L193

Here the model output is manually adapted to avoid collisions based on the estimated occupancy map. You may put it after your RetroMotion decoder. 