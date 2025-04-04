# Copyright (c) 2024 William Ljungbergh
# Copyright (c) 2025 Yinzhe Shen
#
# Changes by Yinzhe Shen on 2025-04-01:
# - Adapt to DMAD

import copy
from dataclasses import dataclass
from typing import List
import uuid
from mmdet3d.datasets.pipelines import Compose
from mmdet3d.models import build_model
from mmdet3d.core.bbox import DepthInstance3DBoxes
from mmcv.runner import load_checkpoint
from mmcv import Config
import numpy as np
from pyquaternion import Quaternion
import torch
from projects.mmdet3d_plugin.uniad.detectors.uniad_e2e import UniAD
from nuscenes.eval.common.utils import (
    quaternion_yaw,
)

NUSCENES_CAM_ORDER = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

# NOTE: this is what they do to the can bus signals
# in preproc of the dataset
### pos 3 | m in global frame
### orientation 4 | quaternion expressed in the ego-vehicle frame
### accel 3 | m/s^2 in ego-vehicle frame
### rotation_rate 3 | rad/s in ego-vehicle frame
### vel 3 | m/s in ego-vehicle frame
### zeros 2

# in preproc of the dataloader
### rotation = Quaternion(input_dict["ego2global_rotation"]) # = nuscenes pose_record["rotation"]
### translation = input_dict["ego2global_translation"] # = nuscenes pose_record["translation"]
### can_bus = input_dict["can_bus"]
### can_bus[:3] = translation
### can_bus[3:7] = rotation
### patch_angle = quaternion_yaw(rotation) / np.pi * 180
### if patch_angle < 0:
###     patch_angle += 360
### can_bus[-2] = patch_angle / 180 * np.pi
### can_bus[-1] = patch_angle

# this means that the can bus signals are:
# 0-3 translation in global frame (to ego-frame)
# 3-7 rotation in global frame (to ego-frame)
# 7-10 acceleration in ego-frame
# 10-13 rotation rate in ego-frame
# 13-16 velocity in ego-frame
# 16 patch angle in degrees
# 17 patch angle in radians


@dataclass
class UniADInferenceInput:
    imgs: np.ndarray
    """shape: (n-cams (6), h (900), w (1600) c (3)) | images without any preprocessing. should be in RGB order as uint8"""
    lidar_pose: np.ndarray
    """shape: (3, 4) | lidar pose in global frame"""
    lidar2img: np.ndarray
    """shape: (n-cams (6), 4, 4) | lidar2img transformation matrix, i.e., lidar2cam @ camera2img"""
    timestamp: float
    """timestamp of the current frame in seconds"""
    can_bus_signals: np.ndarray
    """shape: (16,) | see above for details"""
    command: int
    """0: right, 1: left, 2: straight"""


@dataclass
class UniADAuxOutputs:
    objects_in_bev: np.ndarray  # N x [x, y, width, height, yaw]
    object_classes: List[str]  # (N, )
    object_scores: np.ndarray  # (N, )
    object_ids: np.ndarray  # (N, )
    future_trajs: np.ndarray  # (N, 6 modes, 12 timesteps, [x y])

    def to_json(self) -> dict:
        n_objects = len(self.object_classes)
        return dict(
            objects_in_bev=self.objects_in_bev.tolist() if n_objects > 0 else None,
            object_classes=self.object_classes if n_objects > 0 else None,
            object_scores=self.object_scores.tolist() if n_objects > 0 else None,
            object_ids=self.object_ids.tolist() if n_objects > 0 else None,
            future_trajs=self.future_trajs.tolist() if n_objects > 0 else None,
        )

    @classmethod
    def empty(cls) -> "UniADAuxOutputs":
        return cls(
            objects_in_bev=np.zeros((0, 5)),
            object_classes=[],
            object_scores=np.zeros((0, 1)),
            object_ids=np.zeros((0, 1)),
            future_trajs=np.zeros((0, 6, 12, 2)),
        )


@dataclass
class UniADInferenceOutput:
    trajectory: np.ndarray
    """shape: (n-future (6), 2) | predicted trajectory in the ego-frame @ 2Hz"""
    aux_outputs: UniADAuxOutputs
    """aux outputs such as objects, tracks, segmentation and motion forecast"""


class UniADRunner:
    def __init__(self, config_path: str, checkpoint_path: str, device: torch.device):
        config = Config.fromfile(config_path)
        self.config = config

        self.model: UniAD = build_model(
            config.model, train_cfg=None, test_cfg=config.get("test_cfg")
        )

        self.model.eval()
        # load the checkpoint
        if checkpoint_path is not None:
            ckpt = load_checkpoint(self.model, checkpoint_path, map_location="cpu")
            self.classes = ckpt["meta"]["CLASSES"]
        else:
            raise ValueError("checkpoint_path is None")

        self.model = self.model.to(device)
        self.device = device
        self.preproc_pipeline = Compose(config.inference_pipeline)
        self.reset()

    def reset(self):
        # making a new scene token for each new scene. these are used in the model.
        self.scene_token = str(uuid.uuid4())
        self.prev_frame_info = {
            "prev_bev": None,
            "scene_token": None,
            "prev_pos": 0,
            "prev_angle": 0,
        }

    def _preproc_canbus(self, input: UniADInferenceInput):
        """Preprocesses the raw canbus signals from nuscenes."""
        rotation = Quaternion(input.can_bus_signals[3:7])
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        if patch_angle < 0:
            patch_angle += 360
        # extend the canbus signals with the patch angle, first in radians then in degrees
        input.can_bus_signals = np.append(
            input.can_bus_signals, patch_angle / 180 * np.pi
        )
        input.can_bus_signals = np.append(input.can_bus_signals, patch_angle)
        # UniAD has this, which is faulty as it does not copy the four elements, but just the first one, but we follow it for now
        input.can_bus_signals[3:7] = -rotation

    @torch.no_grad()
    def forward_inference(self, input: UniADInferenceInput) -> UniADInferenceOutput:
        """Run inference without all the preprocessed dataset stuff."""
        # permute rgb -> bgr
        imgs = input.imgs[:, :, :, ::-1]
        # input to preproc shoudl be dict(img=imgs) where imgs: n x h x w x c in bgr format
        preproc_input = dict(img=imgs)
        # run it through the inference pipeline (which is same as eval pipeline except not loading annotations)
        preproc_output = self.preproc_pipeline(preproc_input)
        # collect in array as will convert to tensor, but currently it is a list of arrays (n, h, w, c)
        imgs = np.array(preproc_output["img"])
        # move back to the nchw format
        imgs = np.moveaxis(imgs, -1, 1)
        # convert to tensor and move to device
        imgs = torch.from_numpy(imgs).to(self.device)
        # img should be (1, n, 3, h, w)
        imgs = imgs.unsqueeze(0)
        # move other input to the device as well
        l2g_t = (
            torch.from_numpy(input.lidar_pose[:3, 3]).to(self.device).unsqueeze(0)
        ).float()  # should be 1x3
        l2g_r_mat = (
            torch.from_numpy(input.lidar_pose[:3, :3]).to(self.device).unsqueeze(0)
        ).float()  # should be 1x3x3
        timestamp = (
            torch.from_numpy(np.array([input.timestamp])).to(self.device).unsqueeze(0)
        )
        # we are preproccessing the canbus signals only currently.
        # TODO: fix preproc to include the image preprocessing as well. this is currently done
        # in mmcv (i.e., numpy) and not torch.
        self._preproc_canbus(input)

        # we need to emulate the img_metas here in order to run the model.
        img_metas = [
            {
                "scene_token": self.scene_token,
                "can_bus": input.can_bus_signals,
                "lidar2img": input.lidar2img,  # lidar2cam @ camera2img
                "img_shape": preproc_output["img_shape"],
                # we need this as they are used in the model somewhere.
                "box_type_3d": DepthInstance3DBoxes,
                "pts_filename": "dummy",
            }
        ]

        tmp_pos = copy.deepcopy(img_metas[0]["can_bus"][:3])
        tmp_angle = copy.deepcopy(img_metas[0]["can_bus"][-1])
        # first frame
        if self.prev_frame_info["scene_token"] is None:
            img_metas[0]["can_bus"][:3] = 0
            img_metas[0]["can_bus"][-1] = 0
        # following frames
        else:
            img_metas[0]["can_bus"][:3] -= self.prev_frame_info["prev_pos"]
            img_metas[0]["can_bus"][-1] -= self.prev_frame_info["prev_angle"]

        self.prev_frame_info["prev_pos"] = tmp_pos
        self.prev_frame_info["prev_angle"] = tmp_angle
        self.prev_frame_info["scene_token"] = self.scene_token

        outs_track, outs_seg = self.model.simple_test_track(
            imgs, l2g_t, l2g_r_mat, img_metas, timestamp
        )
        outs_track[0] = self.model.upsample_bev_if_tiny(outs_track[0])

        # get the bev embedding
        bev_embed = outs_track[0]["bev_embed"]

        # get the motion
        traj_output, outs_motion = self.model.motion_head.forward_test(
            bev_embed, outs_track[0], outs_seg[0]
        )
        #  N x 6 modes x 12 timesteps x 5 states (x, y, sigma_x, sigma_y, rho)
        future_trajs = traj_output[0]["traj"][:-1]  # last one is ego
        outs_motion["bev_pos"] = outs_track[0]["bev_pos"]
        # get the occ result
        occ_no_query = outs_motion["track_query"].shape[1] == 0
        if occ_no_query:
            occ_mask = torch.zeros(
                (1, 1 + self.model.occ_head.n_future, 1, *self.model.occ_head.bev_size),
                device=self.device,
            ).long()
        else:
            ins_query = self.model.occ_head.merge_queries(
                outs_motion, self.model.occ_head.detach_query_pos
            )
            _, pred_ins_logits = self.model.occ_head.forward(
                bev_embed, ins_query=ins_query
            )
            pred_ins_logits = pred_ins_logits[:, :, : 1 + self.model.occ_head.n_future]
            pred_ins_sigmoid = pred_ins_logits.sigmoid()
            pred_seg_scores = pred_ins_sigmoid.max(1)[0]
            occ_mask = (
                (pred_seg_scores > self.model.occ_head.test_seg_thresh)
                .long()
                .unsqueeze(2)
            )

        # get the planning output
        outs_planning = self.model.planning_head.forward(
            bev_embed,
            occ_mask,
            outs_motion["bev_pos"],
            outs_motion["sdc_traj_query"],
            outs_motion["sdc_track_query"],
            command=torch.tensor(input.command).to(self.device).unsqueeze(0),
        )

        n_objects = outs_track[0]["boxes_3d"].tensor.shape[0]

        aux_outputs = (
            UniADAuxOutputs(
                objects_in_bev=_format_boxes(outs_track[0]["boxes_3d"]).cpu().numpy(),
                object_scores=outs_track[0]["scores_3d"].cpu().numpy(),
                object_classes=[self.classes[i] for i in outs_track[0]["labels_3d"]],
                object_ids=outs_track[0]["track_ids"].cpu().numpy(),
                future_trajs=_format_trajs(future_trajs[..., :2])
                .cpu()
                .numpy(),  # N x 6 modes x 12 timesteps x 2 states (x y)
            )
            if n_objects > 0
            else UniADAuxOutputs.empty()
        )

        return UniADInferenceOutput(
            trajectory=_format_trajs(outs_planning["sdc_traj"])[0].cpu().numpy(),
            aux_outputs=aux_outputs,
        )


def _format_boxes(boxes: DepthInstance3DBoxes) -> torch.Tensor:
    """Format the boxes into the format defined by the model api.

    This code uses mmdet3d v0.17.1 which have the following coordinate system.

                        up z    y front (yaw=0)
                            ^   ^
                            |  /
                            | /
                            0 ------> x right (yaw=-0.5*pi)


    However, in the API we want the new format defined in mmdet3d >= v.1.x

                            up z    x front (yaw=0)
                            ^   ^
                            |  /
                            | /
    (yaw=0.5pi) left y <--- 0 ----- (yaw=-0.5*pi)

    ref:
    v.0 https://mmdetection3d.readthedocs.io/en/v0.17.1/api.html#mmdet3d.core.bbox.LiDARInstance3DBoxes
    v.1 https://mmdetection3d.readthedocs.io/en/latest/api.html#mmdet3d.structures.bbox_3d.LiDARInstance3DBoxes

    """

    objs = boxes.bev
    # lets change to x-front y-left
    x_ = objs[:, 0].clone()
    y_ = objs[:, 1].clone()
    objs[:, 0] = y_
    objs[:, 1] = -x_

    # lets change the yaw
    objs[:, -1] = -objs[:, -1]
    # and move xywlr -> xylwr
    objs = objs[:, [0, 1, 3, 2, 4]]

    return objs


def _format_trajs(trajs: torch.Tensor) -> torch.Tensor:
    """See above definition of coordinate transforms."""
    x_ = trajs[..., 0].clone()
    y_ = trajs[..., 1].clone()
    trajs[..., 0] = y_
    trajs[..., 1] = -x_
    return trajs


def _get_sample_input(nusc, nusc_can, scene_name, sample) -> UniADInferenceInput:
    timestamp = sample["timestamp"]
    # get the cameras for this sample
    camera_types = [
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_FRONT_LEFT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    ]

    # ego pose via lidar sensor sample data
    lidar_token = sample["data"]["LIDAR_TOP"]
    lidar_sample_data = nusc.get("sample_data", lidar_token)
    ego_pose = nusc.get("ego_pose", lidar_sample_data["ego_pose_token"])
    ego_translation = np.array(ego_pose["translation"])
    ego_rotation_quat = Quaternion(array=ego_pose["rotation"])
    ego2global = np.eye(4)
    ego2global[:3, 3] = ego_translation
    ego2global[:3, :3] = ego_rotation_quat.rotation_matrix

    # get cameras
    camera_tokens = [sample["data"][camera_type] for camera_type in camera_types]
    # sample data for each camera
    camera_sample_data = [nusc.get("sample_data", token) for token in camera_tokens]
    # get the camera calibrations
    camera_calibrations = [
        nusc.get("calibrated_sensor", cam["calibrated_sensor_token"])
        for cam in camera_sample_data
    ]
    # get the image filepaths
    image_filepaths = [
        nusc.get_sample_data(cam_token)[0] for cam_token in camera_tokens
    ]
    # get the camera instrinsics
    cam_instrinsics = [
        np.array(cam_calib["camera_intrinsic"]) for cam_calib in camera_calibrations
    ]
    # compute the camera2img and camera2ego transformations
    camera2img = []
    cam2global = []
    for i in range(len(camera_types)):
        # camera 2 image
        c2i = np.eye(4)
        c2i[:3, :3] = cam_instrinsics[i]
        camera2img.append(c2i)
        # camera 2 ego
        c2e = np.eye(4)
        c2e[:3, 3] = np.array(camera_calibrations[i]["translation"])
        c2e[:3, :3] = Quaternion(
            array=camera_calibrations[i]["rotation"]
        ).rotation_matrix
        # ego 2 global (for camera time)
        cam_e2g = nusc.get("ego_pose", camera_sample_data[i]["ego_pose_token"])
        cam_e2g_t = np.array(cam_e2g["translation"])
        cam_e2g_r = Quaternion(array=cam_e2g["rotation"])
        e2g = np.eye(4)
        e2g[:3, 3] = cam_e2g_t
        e2g[:3, :3] = cam_e2g_r.rotation_matrix
        # cam 2 global
        cam2global.append(e2g @ c2e)

    # load the images in rgb hwc format
    images = []
    for filepath in image_filepaths:
        img = cv2.imread(filepath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
    images = np.array(images)

    # get the lidar calibration
    lidar_sample_data = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    lidar_calibration = nusc.get(
        "calibrated_sensor", lidar_sample_data["calibrated_sensor_token"]
    )
    lidar_translation = np.array(lidar_calibration["translation"])
    lidar_rotation_quat = Quaternion(array=lidar_calibration["rotation"])
    lidar2ego = np.eye(4)
    lidar2ego[:3, 3] = lidar_translation
    lidar2ego[:3, :3] = lidar_rotation_quat.rotation_matrix

    lidar2global = ego2global @ lidar2ego
    global2lidar = np.linalg.inv(lidar2global)
    # the lidar2img should take into consideration that the the timestamps are not the same
    # because of this we go img -> cam -> ego -> global -> ego' -> lidar
    cam2lidar = [global2lidar.copy() @ c2g for c2g in cam2global]
    lidar2cam = [np.linalg.inv(c2l) for c2l in cam2lidar]
    lidar2img = [c2i @ l2c for c2i, l2c in zip(camera2img, lidar2cam)]

    # get the canbus signals
    pose_messages = nusc_can.get_messages(scene_name, "pose")
    can_times = [pose["utime"] for pose in pose_messages]
    assert np.all(np.diff(can_times) > 0), "canbus times not sorted"
    # find the pose that is less than the current timestamp and closest to it
    pose_idx = np.searchsorted(can_times, timestamp)
    # get the canbus signals in the correct order
    canbus_singal_order = ["pos", "orientation", "accel", "rotation_rate", "vel"]
    canbus_signals = np.concatenate(
        [
            np.array(pose_messages[pose_idx][signal_type])
            for signal_type in canbus_singal_order
        ]
    )

    return UniADInferenceInput(
        imgs=images,
        lidar_pose=lidar2global,
        lidar2img=lidar2img,
        timestamp=timestamp,
        can_bus_signals=canbus_signals,
        command=0,  # right
    )


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    runner = UniADRunner(
        config_path="/UniAD/projects/configs/stage2_e2e/inference_e2e.py",
        checkpoint_path="/UniAD/ckpts/uniad_base_e2e.pth",
        device=torch.device(device),
    )

    # only load this for testing
    from nuscenes.nuscenes import NuScenes
    from nuscenes.can_bus.can_bus_api import NuScenesCanBus
    import matplotlib.pyplot as plt
    import cv2

    # load the first surround-cam in nusc mini
    nusc = NuScenes(version="v1.0-mini", dataroot="./data/nuscenes")
    nusc_can = NuScenesCanBus(dataroot="./data/nuscenes")
    scene_name = "scene-0103"
    scene = [s for s in nusc.scene if s["name"] == scene_name][0]
    # get the first sample in the scene
    sample = nusc.get("sample", scene["first_sample_token"])

    for i in range(60):
        inference_input = _get_sample_input(nusc, nusc_can, scene_name, sample)
        if i > 4:
            inference_input.command = 2  # straight
        plan = runner.forward_inference(inference_input)
        # plot in bev
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(inference_input.imgs[0])
        ax[0].axis("off")

        ax[1].plot(plan.trajectory[:, 0], plan.trajectory[:, 1], "r-*")
        ax[1].set_aspect("equal")
        ax[1].set_xlabel("x (m)")
        ax[1].set_ylabel("y (m)")

        # save fig
        fig.savefig(f"{scene_name}_{str(i).zfill(3)}_{sample['timestamp']}.png")
        plt.close(fig)
        if sample["next"] == "":
            break
        sample = nusc.get("sample", sample["next"])
