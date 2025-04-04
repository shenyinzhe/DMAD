# Train/Eval Models
> Modified from UniAD.

## Train <a name="example"></a>
```shell
# stage1:
python -m torch.distributed.launch --nproc_per_node=4 --master_addr="127.0.0.1" --master_port=28593 --nnodes=1 --node_rank=0 tools/train.py projects/configs/stage1_track_map/dmad_stage1.py --launcher pytorch --deterministic --work-dir work_dirs/dmad_stage1

# stage2:
python -m torch.distributed.launch --nproc_per_node=4 --master_addr="127.0.0.1" --master_port=28593 --nnodes=1 --node_rank=0 tools/train.py projects/configs/stage2_e2e/dmad_stage2.py --launcher pytorch --deterministic --work-dir work_dirs/dmad_stage2
```
The first stage takes ~ 42 GB GPU memory, ~ 2 days for 6 epochs on 4 A100 GPUs.

The second stage takes ~ 23 GB GPU memory, ~ 5 days for 20 epochs on 4 A100 GPUs.


## Evaluation <a name="eval"></a>
```shell
# stage1:
python -m torch.distributed.launch --nproc_per_node=1 --master_addr="127.0.0.1" --master_port=28691 --nnodes=1 --node_rank=0 tools/test.py projects/configs/stage1_track_map/dmad_stage1.py  ckpts/dmad_stage1.pth --launcher pytorch --eval box --show-dir work_dirs/eval --out output/results.pkl

# stage2:
python -m torch.distributed.launch --nproc_per_node=1 --master_addr="127.0.0.1" --master_port=28691 --nnodes=1 --node_rank=0 tools/test.py projects/configs/stage2_e2e/dmad_stage2.py  ckpts/dmad_stage2.pth --launcher pytorch --eval box --show-dir work_dirs/eval --out output/results.pkl
```

## Visualization <a name="vis"></a>
```shell
# please refer to  ./tools/uniad_vis_result.sh
python ./tools/analysis_tools/visualize/run.py \
    --predroot /PATH/TO/YOUR/RESULTS.pkl \
    --out_folder /PATH/TO/YOUR/OUTPUT \
    --demo_video test_demo.avi \
    --project_to_cam True
```
---
<- Last Page: [Prepare The Dataset](./DATA_PREP.md)

-> Next Page: [Closed-loop Eval](./EVAL_NEURONCAP.md)