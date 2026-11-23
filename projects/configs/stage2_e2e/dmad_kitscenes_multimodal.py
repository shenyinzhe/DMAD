# Zero-shot open-loop inference of DMAD stage 2 on KITScenes Multimodal:
# images resampled into the virtual nuScenes cameras, causal ego state and the
# 3 s driving command (see docs/KITSCENES.md and
# projects/mmdet3d_plugin/datasets/kitscenes_multimodal_dataset.py). Model and test settings come from dmad_stage2.py.
_base_ = ['./dmad_stage2.py']

ann_file = 'data/kitscenes/multimodal_e2e_dataset.json'  # override with --ann-file

img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
class_names = [
    "car", "truck", "construction_vehicle", "bus", "trailer",
    "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone",
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='WarpMultiViewImageHomography', dst_size=(1600, 900)),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
    dict(type='Collect3D',
         keys=['img', 'timestamp'],
         meta_keys=['box_type_3d', 'ego2global_translation', 'ego2global_rotation',
                    'lidar2img', 'can_bus', 'img_shape', 'l2g_r_mat', 'l2g_t',
                    'scene_token', 'cam_intrinsic', 'command', 'pts_filename',
                    'sample_idx'])
]
_dataset = dict(
    type='KITScenesMultimodalDataset',
    ann_file=ann_file,
    data_root='',
    splits=['val', 'overlap_train_val'],
    frame_gap=5,
    pipeline=test_pipeline,
    test_mode=True,
)
data = dict(test=_dataset, val=_dataset)

# No GT-based evaluation for this dataset.
evaluation = dict(interval=1000, pipeline=test_pipeline, save_best=None)
