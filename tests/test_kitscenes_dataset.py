"""Tests for KITScenesMultimodalDataset — adapter for the KITScenes Multimodal E2E split.

Run inside apptainer (the mmdet3d/mmcv stack lives there). Pytest is not
required; the script is callable as `python tests/test_kitscenes_dataset.py`.
"""
import os
import sys
import json
import tempfile
import unittest

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# We import lazily inside tests so the file at least parses without mmdet3d.

# Default to the dataset JSON from docs/KITSCENES.md; override with
# $KITSCENES_E2E_JSON if it lives elsewhere.
INDEX_JSON = os.environ.get(
    'KITSCENES_E2E_JSON',
    os.path.join(REPO_ROOT, 'data/kitscenes/multimodal_e2e_dataset.json'),
)


def _has_real_data():
    return os.path.isfile(INDEX_JSON)


class TestKITScenesHelpers(unittest.TestCase):
    """Unit tests that don't require mmdet3d (pure-python helpers)."""

    def test_command_mapping(self):
        from projects.mmdet3d_plugin.datasets.kitscenes_multimodal_dataset import HLI_TO_COMMAND
        # 0 = right, 1 = left, 2 = straight (DMAD inherits UniAD's convention)
        self.assertEqual(HLI_TO_COMMAND['turn right'], 0)
        self.assertEqual(HLI_TO_COMMAND['turn left'], 1)
        self.assertEqual(HLI_TO_COMMAND['u-turn'], 1)
        self.assertEqual(HLI_TO_COMMAND['drive straight on'], 2)
        self.assertEqual(HLI_TO_COMMAND['use left lane'], 2)
        self.assertEqual(HLI_TO_COMMAND['use right lane'], 2)
        self.assertEqual(HLI_TO_COMMAND['overtake'], 2)
        # default fallback
        self.assertEqual(HLI_TO_COMMAND.get('', 2), 2)

    def test_camera_mapping(self):
        from projects.mmdet3d_plugin.datasets.kitscenes_multimodal_dataset import CAM_MAPPING
        self.assertEqual(CAM_MAPPING['camera_ring_front'], 'CAM_FRONT')
        self.assertEqual(CAM_MAPPING['camera_ring_front_left'], 'CAM_FRONT_LEFT')
        self.assertEqual(CAM_MAPPING['camera_ring_front_right'], 'CAM_FRONT_RIGHT')
        self.assertEqual(CAM_MAPPING['camera_ring_rear'], 'CAM_BACK')
        self.assertEqual(CAM_MAPPING['camera_ring_rear_left'], 'CAM_BACK_LEFT')
        self.assertEqual(CAM_MAPPING['camera_ring_rear_right'], 'CAM_BACK_RIGHT')

    def test_queue_offsets(self):
        from projects.mmdet3d_plugin.datasets.kitscenes_multimodal_dataset import queue_offsets
        # queue_length=3, frame_gap=5 -> [-10, -5, 0]
        self.assertEqual(queue_offsets(3, 5), [-10, -5, 0])
        # queue_length=1 -> [0]
        self.assertEqual(queue_offsets(1, 5), [0])
        # queue_length=4, frame_gap=2 -> [-6, -4, -2, 0]
        self.assertEqual(queue_offsets(4, 2), [-6, -4, -2, 0])


def _make_synthetic_e2e_dataset(uuid='aaa-bbb-ccc', kf=40):
    """Build a minimal e2e_dataset.json fixture with 1 UUID / 1 keyframe.

    Past covers kf-40..kf, future covers kf+1..kf+50 (translations grow
    linearly so velocity computation is well-defined).
    """
    cameras = {
        'fv_m': 'camera_ring_front',
        'fv_l': 'camera_ring_front_left',
        'fv_r': 'camera_ring_front_right',
        'rv_m': 'camera_ring_rear',
        'rv_l': 'camera_ring_rear_left',
        'rv_r': 'camera_ring_rear_right',
    }
    # A canonical front-camera T_to_reference (cam_z = base_x, etc.)
    R_front = [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ]
    calib = {alias: {
        'K': [[1841.0, 0.0, 1740.0], [0.0, 1841.0, 1139.0], [0.0, 0.0, 1.0]],
        'R': R_front,
        't': [0.2, 0.0, -0.2],
    } for alias in cameras}

    def _pose(i):
        # 1 m/step forward in keyframe-local frame (translation goes
        # negative for past, positive for future; keyframe at origin).
        x = float(i - kf)
        return {
            't_offset_s': round((i - kf) * 0.1, 4),
            'translation': [x, 0.0, 0.0],
            'rotation_wxyz': [1.0, 0.0, 0.0, 0.0],
        }

    def _imgs(i):
        return {alias: f'{uuid}/{cam}/{i:010d}.jpg'
                for alias, cam in cameras.items()}

    traj_past = {str(i): _pose(i) for i in range(kf - 40, kf + 1)}
    traj_future = {str(i): _pose(i) for i in range(kf + 1, kf + 51)}
    images_past = {str(i): _imgs(i) for i in range(kf - 40, kf + 1)}
    images_future = {str(i): _imgs(i) for i in range(kf + 1, kf + 51)}

    return {
        'metadata': {
            'cameras': cameras,
            'past_sec': 4.0,
            'future_sec': 5.0,
            'hz': 10.0,
            'traj_convention': 'local_keyframe_centered',
            'labels_file': 'synthetic',
        },
        uuid: {
            str(kf): {
                'data_root': '/tmp/synthetic_kitscenes',
                'calib': calib,
                'scenario_type': 'synthetic',
                'veh_in_front': False,
                'high_level_instruction': 'drive straight on',
                'high_level_instruction_3s': 'turn left',
                'scenario_type_3s': 'synthetic_3s',
                'split': 'val',
                'traj_past': traj_past,
                'traj_future': traj_future,
                'images_past': images_past,
                'images_future': images_future,
                'n_missing_frames': 0,
            }
        }
    }


class TestKITScenesSynthetic(unittest.TestCase):
    """Unit tests for the e2e_dataset.json codepath using a synthetic fixture
    (no real data required)."""

    def setUp(self):
        from projects.mmdet3d_plugin.datasets.kitscenes_multimodal_dataset import KITScenesMultimodalDataset
        self.uuid = 'aaa-bbb-ccc'
        self.kf = 40
        self.fixture = _make_synthetic_e2e_dataset(self.uuid, self.kf)
        self.tmp = tempfile.NamedTemporaryFile('w', suffix='.json', delete=False)
        json.dump(self.fixture, self.tmp)
        self.tmp.flush()
        self.tmp.close()
        # Build the dataset shell (skip __init__).
        self.ds = KITScenesMultimodalDataset.__new__(KITScenesMultimodalDataset)
        self.ds.queue_length = 3
        self.ds.frame_gap = 5
        self.ds.splits = ['val']

    def tearDown(self):
        os.unlink(self.tmp.name)

    def test_format_detection_dispatches_to_e2e_loader(self):
        infos = self.ds.load_annotations(self.tmp.name)
        # queue_length=3 -> 3 samples per keyframe
        self.assertEqual(len(infos), 3)
        for s in infos:
            self.assertEqual(s['uuid'], self.uuid)
            self.assertEqual(s['keyframe'], self.kf)
            self.assertEqual(s['scene_token'], f'{self.uuid}_{self.kf:010d}')

    def test_queue_indices_are_keyframe_minus_offsets(self):
        infos = self.ds.load_annotations(self.tmp.name)
        # offsets [-10, -5, 0] from kf=40 -> indices [30, 35, 40]
        self.assertEqual([s['sample_idx'] for s in infos], [30, 35, 40])
        self.assertTrue(infos[-1]['is_current'])
        self.assertTrue(infos[0]['is_first_in_scene'])

    def test_image_paths_resolve_under_data_root(self):
        infos = self.ds.load_annotations(self.tmp.name)
        kf_info = [s for s in infos if s['is_current']][0]
        # CAM_FRONT maps to fv_m which is camera_ring_front
        front = kf_info['cams']['CAM_FRONT']
        self.assertTrue(front['data_path'].startswith('/tmp/synthetic_kitscenes/'))
        self.assertIn('camera_ring_front', front['data_path'])
        self.assertIn(f'{self.kf:010d}.jpg', front['data_path'])

    def test_virtual_nuscenes_camera(self):
        from projects.mmdet3d_plugin.datasets.kitscenes_multimodal_dataset import (
            nuscenes_intrinsic, nuscenes_rotation, _orthogonalize)
        infos = self.ds.load_annotations(self.tmp.name)
        cam = infos[-1]['cams']['CAM_FRONT']
        # intrinsics/rotation are the virtual nuScenes camera, not the KITScenes calib
        np.testing.assert_allclose(cam['cam_intrinsic'], nuscenes_intrinsic('CAM_FRONT'), rtol=1e-6)
        np.testing.assert_allclose(cam['sensor2ego_rotation_matrix'], nuscenes_rotation('CAM_FRONT'))
        # the warp maps a dst pixel to the src pixel of the same viewing ray
        calib = self.fixture[self.uuid][str(self.kf)]['calib']['fv_m']
        src_K = np.asarray(calib['K']); src_R = _orthogonalize(calib['R'])
        dst_K = nuscenes_intrinsic('CAM_FRONT'); dst_R = nuscenes_rotation('CAM_FRONT')
        ray = np.array([0.3, -0.1, 1.0])  # camera-frame ray of the virtual camera
        dst_px = dst_K @ ray
        src_px = src_K @ src_R.T @ dst_R @ ray
        mapped = cam['src_from_dst'] @ dst_px
        np.testing.assert_allclose(mapped[:2] / mapped[2], src_px[:2] / src_px[2], rtol=1e-9)

    def test_causal_speed_ignores_future(self):
        infos = self.ds.load_annotations(self.tmp.name)
        # 1 m per 0.1 s frame forward -> 10 m/s, no acceleration
        for s in infos:
            self.assertAlmostEqual(s['can_bus'][13], 10.0)
            self.assertAlmostEqual(s['can_bus'][7], 0.0)
        # changing the future trajectory must not change any input
        for f, pose in self.fixture[self.uuid][str(self.kf)]['traj_future'].items():
            pose['translation'] = [100.0, 50.0, 0.0]
        with open(self.tmp.name, 'w') as fh:
            json.dump(self.fixture, fh)
        infos2 = self.ds.load_annotations(self.tmp.name)
        for a, b in zip(infos, infos2):
            np.testing.assert_array_equal(a['can_bus'], b['can_bus'])

    def test_can_bus_is_18_dim(self):
        infos = self.ds.load_annotations(self.tmp.name)
        for s in infos:
            self.assertEqual(s['can_bus'].shape, (18,))

    def test_command_text_is_3s_label(self):
        infos = self.ds.load_annotations(self.tmp.name)
        for s in infos:
            self.assertEqual(s['command_text'], 'turn left')
            self.assertEqual(s['scenario_type'], 'synthetic_3s')

    def test_missing_3s_label_skips_keyframe(self):
        del self.fixture[self.uuid][str(self.kf)]['high_level_instruction_3s']
        with open(self.tmp.name, 'w') as f:
            json.dump(self.fixture, f)
        self.assertEqual(len(self.ds.load_annotations(self.tmp.name)), 0)

    def test_split_filter_excludes_other_splits(self):
        # Flip split to 'train' and verify it gets filtered out under val.
        self.fixture[self.uuid][str(self.kf)]['split'] = 'train'
        with open(self.tmp.name, 'w') as f:
            json.dump(self.fixture, f)
        infos = self.ds.load_annotations(self.tmp.name)
        self.assertEqual(len(infos), 0)


@unittest.skipUnless(_has_real_data(),
                     "requires e2e_dataset.json (set $KITSCENES_E2E_JSON, "
                     "see docs/KITSCENES.md)")
class TestKITScenesRealData(unittest.TestCase):
    """Integration tests that load a real e2e_dataset.json."""

    def _build_ds(self):
        from projects.mmdet3d_plugin.datasets.kitscenes_multimodal_dataset import KITScenesMultimodalDataset
        ds = KITScenesMultimodalDataset.__new__(KITScenesMultimodalDataset)
        ds.queue_length = 3
        ds.frame_gap = 5
        ds.splits = ['val']
        return ds

    def test_load_annotations_val(self):
        ds = self._build_ds()
        infos = ds.load_annotations(INDEX_JSON)
        self.assertGreater(len(infos), 0)
        sample = infos[0]
        for k in ('token', 'scene_token', 'sample_idx', 'cams', 'can_bus',
                  'ego2global_translation', 'ego2global_rotation', 'video_name',
                  'keyframe', 'uuid'):
            self.assertIn(k, sample, f'missing key {k}')
        # scene_token = <uuid>_<keyframe:010d>
        self.assertTrue(sample['scene_token'].split('_')[-1].isdigit())
        # cams keys are NuScenes-style names
        for cam_name in sample['cams']:
            self.assertTrue(cam_name.startswith('CAM_'))
        # data_path actually exists for current frame
        currents = [s for s in infos if s['is_current']]
        self.assertGreater(len(currents), 0)
        for cam_name, cam_info in currents[0]['cams'].items():
            self.assertTrue(os.path.isfile(cam_info['data_path']),
                            f'missing img: {cam_info["data_path"]}')

    def test_high_level_instruction_loaded(self):
        """HLI must be carried through to every sample's command_text."""
        ds = self._build_ds()
        infos = ds.load_annotations(INDEX_JSON)
        self.assertGreater(len(infos), 0)
        with_hli = sum(1 for s in infos if s.get('command_text'))
        self.assertGreater(with_hli, 0.9 * len(infos),
                           f'only {with_hli}/{len(infos)} samples have HLI')
        from projects.mmdet3d_plugin.datasets.kitscenes_multimodal_dataset import HLI_TO_COMMAND
        seen = {s['command_text'] for s in infos if s.get('command_text')}
        seen_norm = {s.strip().lower() for s in seen}
        self.assertTrue(seen_norm.issubset(set(HLI_TO_COMMAND.keys())),
                        f'unexpected HLIs: {seen_norm - set(HLI_TO_COMMAND.keys())}')

    def test_queue_grouping(self):
        ds = self._build_ds()
        infos = ds.load_annotations(INDEX_JSON)

        from collections import Counter
        per_scene = Counter(s['scene_token'] for s in infos)
        # Each scenario should produce exactly queue_length samples
        for scene, n in per_scene.items():
            self.assertEqual(n, 3, f'scene {scene} produced {n} samples (expected 3)')

        # Within a scene the timestamps should be strictly increasing
        # and the LAST sample must be the current frame.
        seen = set()
        for s in infos:
            tok = s['scene_token']
            if tok in seen:
                continue
            seen.add(tok)
            scene_samples = [x for x in infos if x['scene_token'] == tok]
            ts = [x['timestamp'] for x in scene_samples]
            self.assertEqual(ts, sorted(ts), f'{tok}: timestamps not sorted')
            self.assertTrue(scene_samples[-1]['is_current'],
                            f'{tok}: last sample is not the "current" frame')


class TestInferenceScript(unittest.TestCase):
    """The inference script's helpers, tested in isolation."""

    def test_output_filename_for_info(self):
        from tools.kitscenes.inference import output_filename_for_info
        info = {
            'uuid': '00fc1b9b-005e-554c-7733-c4c0adeca015',
            'keyframe': 40,
        }
        self.assertEqual(
            output_filename_for_info(info),
            'pred_00fc1b9b-005e-554c-7733-c4c0adeca015_kf0000000040.npy',
        )

    def test_shard_data_infos_keeps_queue_grouped(self):
        """Sharding must preserve the per-clip 3-frame queue contiguously
        so DMAD's prev_bev accumulation stays intact."""
        from tools.kitscenes.inference import shard_data_infos
        # Build 6 fake clips x 3 frames each.
        infos = []
        for clip in range(6):
            for q in range(3):
                infos.append({
                    'scene_token': f'clip_{clip:02d}',
                    'is_first_in_scene': (q == 0),
                    'is_current': (q == 2),
                })
        # 3-way shard
        s0 = shard_data_infos(infos, shard=0, num_shards=3)
        s1 = shard_data_infos(infos, shard=1, num_shards=3)
        s2 = shard_data_infos(infos, shard=2, num_shards=3)
        # Together they cover everything, no overlap
        self.assertEqual(len(s0) + len(s1) + len(s2), len(infos))
        all_tokens = {x['scene_token'] for x in s0 + s1 + s2}
        self.assertEqual(len(all_tokens), 6)
        # Each shard's clips should each contain exactly 3 contiguous frames
        for shard_infos in (s0, s1, s2):
            from collections import Counter
            c = Counter(x['scene_token'] for x in shard_infos)
            for tok, n in c.items():
                self.assertEqual(n, 3, f'{tok} in shard has {n} frames')
            # is_first_in_scene markers stay aligned
            for i, x in enumerate(shard_infos):
                if i == 0 or shard_infos[i - 1]['scene_token'] != x['scene_token']:
                    self.assertTrue(x['is_first_in_scene'])

    def test_shard_data_infos_single_shard_is_identity(self):
        from tools.kitscenes.inference import shard_data_infos
        infos = [{'scene_token': 'a', 'is_first_in_scene': True},
                 {'scene_token': 'a', 'is_first_in_scene': False},
                 {'scene_token': 'b', 'is_first_in_scene': True}]
        self.assertEqual(shard_data_infos(infos, 0, 1), infos)

    def test_visualizer_does_not_mirror_left_right(self):
        """Regression: a trajectory with col0=+1 (right) must project to
        u > principal_point_u; col0=-1 (left) must project to u < cu.

        Uses an e2e_dataset.json-shaped calib block (alias-keyed `K/R/t`).
        The chosen R is the canonical front-camera mapping cam → base:
          cam Z-fwd  -> base +X
          cam X-right -> base -Y
          cam Y-down  -> base -Z
        """
        from tools.kitscenes.visualize import lidar2img_for_front, project_traj
        calib = {'fv_m': {
            'K': [[1000.0, 0.0, 800.0], [0.0, 1000.0, 450.0], [0.0, 0.0, 1.0]],
            'R': [
                [0.0,  0.0, 1.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ],
            't': [0.0, 0.0, 0.0],
            'resolution': {'width': 1600, 'height': 900},
        }}
        lidar2img, _ = lidar2img_for_front(calib, img_w=1600, img_h=900)

        # A point 1m to the RIGHT, 10m forward, on the ground plane.
        traj_right = np.array([[+1.0, 10.0]], dtype=np.float32)  # (right, fwd)
        uv_r, mask_r = project_traj(traj_right, lidar2img, ground_z=0.0)
        self.assertTrue(mask_r[0], 'point must be in front of camera')
        self.assertGreater(uv_r[0, 0], 800,
                           f'right traj should project to u>cu=800, got u={uv_r[0,0]}')

        traj_left = np.array([[-1.0, 10.0]], dtype=np.float32)
        uv_l, mask_l = project_traj(traj_left, lidar2img, ground_z=0.0)
        self.assertTrue(mask_l[0])
        self.assertLess(uv_l[0, 0], 800,
                        f'left traj should project to u<cu=800, got u={uv_l[0,0]}')

    def test_resolve_shards_from_env(self):
        from tools.kitscenes.inference import resolve_shards_from_env
        # 1 GPU -> no fork
        self.assertEqual(resolve_shards_from_env('0'), ['0'])
        # 4 GPUs
        self.assertEqual(resolve_shards_from_env('0,1,2,3'),
                         ['0', '1', '2', '3'])
        # whitespace tolerated
        self.assertEqual(resolve_shards_from_env(' 2 , 5 '), ['2', '5'])
        # empty / unset -> no sharding (run inline on default device)
        self.assertEqual(resolve_shards_from_env(''), [])
        self.assertEqual(resolve_shards_from_env(None), [])


@unittest.skipUnless(_has_real_data(), "requires the KITScenes data (see docs/KITSCENES.md)")
class TestKITScenesConfig(unittest.TestCase):
    """Smoke-test the inference config end-to-end via the mmdet3d pipeline."""

    CONFIG = os.path.join(REPO_ROOT, 'projects/configs/stage2_e2e/dmad_kitscenes_multimodal.py')

    def test_build_dataset_from_config(self):
        self.assertTrue(os.path.isfile(self.CONFIG),
                        f'config not found: {self.CONFIG}')
        from mmcv import Config
        from mmdet3d.datasets import build_dataset
        import projects.mmdet3d_plugin  # noqa: F401  (registers KITScenesMultimodalDataset)
        cfg = Config.fromfile(self.CONFIG)
        cfg.data.test.test_mode = True
        ds = build_dataset(cfg.data.test)
        self.assertGreater(len(ds), 0)
        sample = ds[0]
        # The default DMAD test pipeline produces 'img', 'img_metas'
        self.assertIn('img', sample)
        self.assertIn('img_metas', sample)
        # six views resampled to 1600x900, padded to 1600x928
        img = sample['img'][0] if isinstance(sample['img'], list) else sample['img']
        self.assertEqual(tuple(img.data.shape), (6, 3, 928, 1600))


if __name__ == '__main__':
    unittest.main(verbosity=2)
