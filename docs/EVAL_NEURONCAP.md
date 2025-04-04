# NeuroNCAP
This repository supports evaluation on [NeuroNCAP](https://github.com/atonderski/neuro-ncap/).

- Prepare model weights and anchors
```shell
wget https://github.com/shenyinzhe/DMAD/releases/download/v1.0/dmad_stage2.pth
wget https://github.com/shenyinzhe/DMAD/releases/download/v1.0/motion_anchor_infos_mode6.pkl
```

- Build Docker image or sif file
```shell
docker build -t dmad:latest -f docker/Dockerfile .
singularity build dmad.sif docker-daemon://dmad:latest
```

- Folder structure looks like this:
```
├── DMAD
|   ├── ckpts
│   |   ├── dmad_stage2.pth
|   ├── data
│   |   ├── others
│   |   |   ├── motion_anchor_infos_mode6.pkl
│   ├── dmad.sif (if running using singularity)
│   ├── ...
```

- Refer to the official [how-to-run](https://github.com/atonderski/neuro-ncap/blob/main/docs/how-to-run.md) for furhter steps.

<- Last Page: [Train/Eval](./TRAIN_EVAL.md)