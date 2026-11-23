"""KITScenes Multimodal adapter for UniAD/DMAD inference.

KITScenes has two splits, Multimodal and LongTail; this adapter reads the
Multimodal E2E dataset definition (JSON, see `docs/KITSCENES.md`). Each
keyframe entry holds `data_root` (images under `<data_root>/<uuid>/camera_ring_*/`),
per-camera `calib` (alias -> {K, R, t}, R/t map camera to ego), `traj_past` /
`traj_future` (keyframe-local frame, 10 Hz), `images_past` and the
driving-command labels.

Input conventions:

1. Cameras: every ring image is resampled into the matching *virtual nuScenes
   camera* (median nuScenes v1.0-mini intrinsics at 1600x900 and camera->ego
   rotation). KITScenes ring cameras are ~87 deg wide versus nuScenes' ~64 deg
   (CAM_BACK ~89 deg), so a plain resize would show objects at ~0.6x their
   trained size. The warp is a pure rotation + intrinsic change,
   `src_from_dst = K_src R_src^T R_nus K_nus^-1`; the camera translation stays
   the KITScenes one (~0.1 m parallax ignored).
2. Ego state: causal (backward) finite differences at 10 Hz for velocity,
   acceleration and angular rate, rotated into the vehicle frame and written to
   the nuScenes CAN-bus slots (accel 7:10 with gravity 9.807 in z, angular rate
   10:13, velocity 13:16). Ego z is set to 0. Only past poses are read.
3. Driving command from the 3 s label window (`high_level_instruction_3s`),
   matching the 6 x 0.5 s = 3 s planning horizon. Text -> command:
   turn right=0, turn left/u-turn=1, everything else=2 (nuScenes convention).

Queue: queue_length=3 with frame_gap=5 -> frames at offsets [-10, -5, 0] from
the keyframe; the keyframe is the current frame.
"""
import json
import math

import cv2
import mmcv
import numpy as np
from pyquaternion import Quaternion

from mmdet.datasets import DATASETS
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import Compose
from mmdet3d.core.bbox import LiDARInstance3DBoxes, Box3DMode

from .nuscenes_e2e_dataset import NuScenesE2EDataset


CAM_MAPPING = {
    'camera_ring_front': 'CAM_FRONT',
    'camera_ring_front_left': 'CAM_FRONT_LEFT',
    'camera_ring_front_right': 'CAM_FRONT_RIGHT',
    'camera_ring_rear': 'CAM_BACK',
    'camera_ring_rear_left': 'CAM_BACK_LEFT',
    'camera_ring_rear_right': 'CAM_BACK_RIGHT',
}

# UniAD/DMAD command codes: 0=Right, 1=Left, 2=Straight/Follow.
HLI_TO_COMMAND = {
    'turn right': 0,
    'turn left': 1,
    'u-turn': 1,
    'drive straight on': 2,
    'use left lane': 2,
    'use right lane': 2,
    'use middle lane': 2,
    'overtake': 2,
    '': 2,
}
COMMAND_FIELD = 'high_level_instruction_3s'

NUSCENES_SIZE = (1600, 900)  # (width, height)
# (fx=fy, cx, cy) at 1600x900 and camera->ego rotation; median over the 10 logs of
# nuScenes v1.0-mini calibrated_sensor.json.
NUSCENES_CAMERAS = {
    'CAM_FRONT': ((1266.417, 816.267, 491.507),
                  [[+0.007493, -0.000001, +0.999972],
                   [-0.999962, +0.004417, +0.007493],
                   [-0.004417, -0.999990, +0.000032]]),
    'CAM_FRONT_LEFT': ((1272.598, 826.615, 479.752),
                       [[+0.821473, +0.002391, +0.570243],
                        [-0.570240, +0.008503, +0.821434],
                        [-0.002884, -0.999961, +0.008349]]),
    'CAM_FRONT_RIGHT': ((1260.847, 807.968, 495.334),
                        [[-0.837405, +0.006608, +0.546543],
                         [-0.546457, +0.011317, -0.837411],
                         [-0.011718, -0.999914, -0.005866]]),
    'CAM_BACK': ((809.221, 829.220, 481.778),
                 [[+0.005157, -0.012785, -0.999905],
                  [+0.999985, +0.002138, +0.005130],
                  [+0.002073, -0.999916, +0.012796]]),
    'CAM_BACK_LEFT': ((1256.741, 792.113, 492.776),
                      [[+0.947821, +0.001629, -0.318799],
                       [+0.318801, -0.000871, +0.947821],
                       [+0.001266, -0.999998, -0.001345]]),
    'CAM_BACK_RIGHT': ((1259.514, 807.253, 501.196),
                       [[-0.930481, +0.008568, -0.366240],
                        [+0.366270, +0.002178, -0.930506],
                        [-0.007175, -0.999961, -0.005164]]),
}
GRAVITY = 9.807
DT = 0.1  # 10 Hz source


# --- small pure helpers (unit-tested) -------------------------------------

def queue_offsets(queue_length, frame_gap):
    """Ascending frame offsets ending at 0, e.g. (3, 5) -> [-10, -5, 0]."""
    return [(i - (queue_length - 1)) * frame_gap for i in range(queue_length)]


def _orthogonalize(R):
    """Project a near-orthogonal 3x3 matrix onto SO(3) via SVD.

    The stored R matrices are float32 with tiny non-orthogonality, which
    pyquaternion's strict matrix->quat conversion rejects.
    """
    R = np.asarray(R, dtype=np.float64)
    U, _, Vt = np.linalg.svd(R)
    R_ortho = U @ Vt
    if np.linalg.det(R_ortho) < 0:
        U[:, -1] *= -1
        R_ortho = U @ Vt
    return R_ortho


def nuscenes_intrinsic(camera):
    (focal, cx, cy), _ = NUSCENES_CAMERAS[camera]
    return np.array([[focal, 0., cx], [0., focal, cy], [0., 0., 1.]])


def nuscenes_rotation(camera):
    return _orthogonalize(np.array(NUSCENES_CAMERAS[camera][1], dtype=np.float64))


def src_from_dst_homography(src_K, src_R, dst_K, dst_R):
    """Pixel map dst -> src for a camera that only changes rotation and intrinsics."""
    return src_K @ src_R.T @ dst_R @ np.linalg.inv(dst_K)


def euler_angles(rotation_wxyz):
    """(roll, pitch, yaw) of a wxyz quaternion."""
    w, x, y, z = rotation_wxyz
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x ** 2 + y ** 2))
    pitch = math.asin(float(np.clip(2 * (w * y - z * x), -1, 1)))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y ** 2 + z ** 2))
    return np.array([roll, pitch, yaw])


def causal_kinematics(poses, frame):
    """Velocity, acceleration (xy, keyframe-local frame) and angular rate at `frame`.

    `poses` maps frame index -> pose dict. Backward differences; falls back to a
    forward difference only when no past pose exists (never at a Multimodal keyframe).
    """
    def xy(f):
        return np.asarray(poses[f]['translation'][:2], dtype=np.float64)

    def eul(f):
        return euler_angles(poses[f]['rotation_wxyz'])

    velocity = np.zeros(2)
    acceleration = np.zeros(2)
    angular_rate = np.zeros(3)
    if frame - 1 in poses:
        velocity = (xy(frame) - xy(frame - 1)) / DT
        angular_rate = (eul(frame) - eul(frame - 1)) / DT
        if frame - 2 in poses:
            previous_velocity = (xy(frame - 1) - xy(frame - 2)) / DT
            acceleration = (velocity - previous_velocity) / DT
    elif frame + 1 in poses:
        velocity = (xy(frame + 1) - xy(frame)) / DT
        angular_rate = (eul(frame + 1) - eul(frame)) / DT
        if frame + 2 in poses:
            next_velocity = (xy(frame + 2) - xy(frame + 1)) / DT
            acceleration = (next_velocity - velocity) / DT
    return velocity, acceleration, angular_rate


# --- the dataset -----------------------------------------------------------

@DATASETS.register_module()
class KITScenesMultimodalDataset(NuScenesE2EDataset):
    """Inference-only KITScenes Multimodal dataset in UniAD/DMAD input format."""

    def __init__(self,
                 ann_file,
                 data_root='',
                 splits=('val', 'overlap_train_val'),
                 frame_gap=5,
                 **kwargs):
        # Bypass NuScenesE2EDataset.__init__, which loads the nuScenes DB.
        self.ann_file = ann_file
        self.data_root = data_root
        self.splits = list(splits)
        self.frame_gap = int(frame_gap)

        self.test_mode = kwargs.get('test_mode', True)
        self.is_debug = False
        self.box_type_3d = LiDARInstance3DBoxes
        self.box_mode_3d = Box3DMode.LIDAR
        self.filter_empty_gt = False

        if 'pipeline' in kwargs:
            self.pipeline = Compose(kwargs['pipeline'])

        self.queue_length = kwargs.get('queue_length', 3)
        self.overlap_test = kwargs.get('overlap_test', False)
        self.bev_size = kwargs.get('bev_size', (200, 200))
        self.predict_steps = kwargs.get('predict_steps', 12)
        self.planning_steps = kwargs.get('planning_steps', 6)
        self.past_steps = kwargs.get('past_steps', 4)
        self.fut_steps = kwargs.get('fut_steps', 4)
        self.eval_mod = kwargs.get('eval_mod', None)

        self.data_infos = self.load_annotations(ann_file)

    # ----- annotation loading -------------------------------------------

    def load_annotations(self, ann_file):
        with open(ann_file, 'r') as f:
            data = json.load(f)
        if not (isinstance(data, dict)
                and isinstance(data.get('metadata'), dict)
                and data['metadata'].get('traj_convention') == 'local_keyframe_centered'):
            raise ValueError(
                f'{ann_file}: not a KITScenes e2e_dataset JSON '
                '(expected metadata.traj_convention == "local_keyframe_centered")')

        cameras = data['metadata'].get('cameras', {})  # alias -> camera_ring_X
        alias_to_nusc = {alias: CAM_MAPPING[cam_dir]
                         for alias, cam_dir in cameras.items()
                         if cam_dir in CAM_MAPPING}
        wanted = set(self.splits)

        data_infos = []
        n_keyframes = 0
        for uuid in sorted(k for k in data.keys() if k != 'metadata'):
            uuid_entry = data[uuid]
            if not isinstance(uuid_entry, dict):
                continue
            for kf_str in sorted(uuid_entry.keys(), key=int):
                entry = uuid_entry[kf_str]
                if entry.get('split') not in wanted:
                    continue
                try:
                    infos = self._build_keyframe_infos(uuid, int(kf_str), entry, alias_to_nusc)
                except Exception as e:
                    print(f'[KITScenesMultimodalDataset] skipping {uuid} kf={kf_str}: {e}')
                    continue
                data_infos.extend(infos)
                n_keyframes += 1
        print(f'[KITScenesMultimodalDataset] loaded {len(data_infos)} samples from '
              f'{n_keyframes} keyframes (splits={self.splits})')
        return data_infos

    def _build_keyframe_infos(self, uuid, kf, entry, alias_to_nusc):
        """queue_length sample dicts for one keyframe."""
        data_root = entry['data_root']
        traj_past = entry.get('traj_past', {})
        images_past = entry.get('images_past', {})
        scene_token = f'{uuid}_{kf:010d}'
        if COMMAND_FIELD not in entry:
            raise KeyError(f'{COMMAND_FIELD} missing')
        hli = entry[COMMAND_FIELD]

        cams_template = {}
        for alias, cam_calib in entry.get('calib', {}).items():
            nusc_name = alias_to_nusc.get(alias)
            if nusc_name is None:
                continue
            src_K = np.asarray(cam_calib['K'], dtype=np.float64)
            src_R = _orthogonalize(np.asarray(cam_calib['R'], dtype=np.float64))
            K = nuscenes_intrinsic(nusc_name)
            R = nuscenes_rotation(nusc_name)
            cams_template[nusc_name] = {
                'alias': alias,
                'sensor2ego_translation': np.asarray(cam_calib['t'], dtype=np.float64),
                'sensor2ego_rotation_matrix': R,
                'cam_intrinsic': K.astype(np.float32),
                'src_from_dst': src_from_dst_homography(src_K, src_R, K, R),
            }

        # Only past poses are used: no future pose enters any input.
        poses = {int(f): p for f, p in traj_past.items()}
        if kf not in poses:
            raise ValueError(f'keyframe {kf} missing from traj_past')

        infos = []
        prev_token = ''
        for qi, fi in enumerate(kf + o for o in queue_offsets(self.queue_length, self.frame_gap)):
            src_fi = fi if fi in poses else kf  # fall back to the keyframe if history is short
            pose = poses[src_fi]
            translation = [float(pose['translation'][0]), float(pose['translation'][1]), 0.0]
            rotation = list(pose['rotation_wxyz'])
            timestamp = float(pose['t_offset_s'])

            velocity, acceleration, angular_rate = causal_kinematics(poses, src_fi)
            yaw = euler_angles(rotation)[2]
            to_vehicle = np.array([[math.cos(yaw), math.sin(yaw)],
                                   [-math.sin(yaw), math.cos(yaw)]])
            can_bus = np.zeros(18, dtype=np.float64)
            can_bus[0:3] = translation
            can_bus[3:7] = rotation
            can_bus[7:9] = to_vehicle @ acceleration
            can_bus[9] = GRAVITY
            can_bus[10:13] = angular_rate
            can_bus[13:15] = to_vehicle @ velocity
            patch_angle = (Quaternion(rotation).yaw_pitch_roll[0] / np.pi * 180.0) % 360.0
            can_bus[-2] = patch_angle / 180.0 * np.pi
            can_bus[-1] = patch_angle

            img_entry = images_past.get(str(src_fi), {})
            cams = {}
            for nusc_name, base in cams_template.items():
                rel = img_entry.get(base['alias'])
                if not rel:
                    continue
                cams[nusc_name] = dict(
                    data_path=f'{data_root}/{rel}',
                    type=nusc_name,
                    sensor2ego_translation=base['sensor2ego_translation'],
                    sensor2ego_rotation_matrix=base['sensor2ego_rotation_matrix'],
                    cam_intrinsic=base['cam_intrinsic'],
                    src_from_dst=base['src_from_dst'],
                )

            token = f'{scene_token}_{fi}'
            infos.append(dict(
                lidar_path='dummy_lidar.pcd',
                token=token,
                sample_idx=fi,
                prev=prev_token,
                next='',
                scene_token=scene_token,
                video_name=f'{uuid}_kf{kf:010d}',
                uuid=uuid,
                keyframe=kf,
                timestamp=timestamp,
                cams=cams,
                can_bus=can_bus,
                ego2global_translation=translation,
                ego2global_rotation=rotation,
                command_text=hli,
                scenario_type=entry.get('scenario_type_3s', ''),
                veh_in_front=entry.get('veh_in_front', None),
                is_current=(fi == kf),
                is_first_in_scene=(qi == 0),
            ))
            prev_token = token
        for i in range(len(infos) - 1):
            infos[i]['next'] = infos[i + 1]['token']
        return infos

    # ----- per-sample dict for the pipeline ------------------------------

    def get_data_info(self, index):
        info = self.data_infos[index]
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=[],
            timestamp=info['timestamp'],
            scene_token=info['scene_token'],
            can_bus=info['can_bus'],
            ego2global_translation=info['ego2global_translation'],
            ego2global_rotation=info['ego2global_rotation'],
        )
        # nuScenes camera order of the UniAD/DMAD pipeline.
        sensors = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                   'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
        # nuScenes lidar (x right, y forward, z up) -> ego (x forward, y left, z up)
        nusc2base = np.array([[0., 1., 0., 0.],
                              [-1., 0., 0., 0.],
                              [0., 0., 1., 0.],
                              [0., 0., 0., 1.]], dtype=np.float64)
        img_paths, lidar2img_rts, lidar2cam_rts, cam_intrinsics, warps = [], [], [], [], []
        for sensor in sensors:
            if sensor not in info['cams']:
                continue
            cam = info['cams'][sensor]
            img_paths.append(cam['data_path'])
            K = np.asarray(cam['cam_intrinsic'], dtype=np.float64)
            T_cam_to_base = np.eye(4)
            T_cam_to_base[:3, :3] = cam['sensor2ego_rotation_matrix']
            T_cam_to_base[:3, 3] = cam['sensor2ego_translation']
            lidar2cam = np.linalg.inv(T_cam_to_base) @ nusc2base
            viewpad = np.eye(4)
            viewpad[:3, :3] = K
            lidar2img_rts.append((viewpad @ lidar2cam).astype(np.float32))
            lidar2cam_rts.append(lidar2cam.astype(np.float32))
            cam_intrinsics.append(K.astype(np.float32))
            warps.append(cam['src_from_dst'])

        cmd_text = str(info.get('command_text', '') or '').strip().lower()
        input_dict.update(dict(
            img_filename=img_paths,
            lidar2img=lidar2img_rts,
            lidar2cam=lidar2cam_rts,
            cam_intrinsic=cam_intrinsics,
            command=np.array(HLI_TO_COMMAND.get(cmd_text, 2)),
            src_from_dst=warps,
        ))
        return input_dict

    def __getitem__(self, idx):
        return self.prepare_test_data(idx)

    def prepare_test_data(self, index):
        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        if 'img_metas' in example and not isinstance(example['img_metas'].data, list):
            example['img_metas'] = mmcv.parallel.DataContainer(
                [example['img_metas'].data], cpu_only=True)
        return example

    def get_cat_ids(self, idx):
        return []


@PIPELINES.register_module()
class WarpMultiViewImageHomography(object):
    """Resample each view with `results['src_from_dst']` (dst pixel -> src pixel)
    into a `dst_size` image. Pixels outside the source image are black (0)."""

    def __init__(self, dst_size=NUSCENES_SIZE):
        self.dst_size = tuple(dst_size)

    def __call__(self, results):
        warps = results['src_from_dst']
        assert len(warps) == len(results['img'])
        imgs = [cv2.warpPerspective(
                    img, np.asarray(warp, dtype=np.float64), self.dst_size,
                    flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                for img, warp in zip(results['img'], warps)]
        results['img'] = imgs
        results['img_shape'] = [im.shape for im in imgs]
        results['ori_shape'] = [im.shape for im in imgs]
        results['pad_shape'] = [im.shape for im in imgs]
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(dst_size={self.dst_size})'
