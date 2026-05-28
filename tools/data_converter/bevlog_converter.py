import os
from os import path as osp
import mmcv
import numpy as np
import argparse
from PIL import Image

"""Simple converter for BEVLOG raw recordings into nuScenes-style infos.

This produces `nuscenes_infos_train.pkl` and `nuscenes_infos_val.pkl`
under the dataset root. The produced infos are minimal: they contain
per-sample `token`, `timestamp`, `lidar_path` (if available) and a
`cams` dict mapping the expected 6 camera keys to image file paths.

This is a pragmatic converter to make the dataset visible to the
BEVMap training pipeline. Annotations are left empty (no GT boxes).
"""


CAM_KEY_ORDER = [
    'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
    'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
]

RAW_CAM_FOLDERS = {
    'fisheye_left': 'CAM_FRONT_LEFT',
    'fisheye_right': 'CAM_FRONT_RIGHT',
    'zed_left': 'CAM_BACK_LEFT',
    'zed_right': 'CAM_BACK_RIGHT'
}


def _collect_frame_list(root_path):
    fl = osp.join(root_path, 'annotations', 'pointclouds', 'frame_list.txt')
    if not osp.exists(fl):
        raise FileNotFoundError(f'frame_list.txt not found at {fl}')
    lines = open(fl, 'r').read().splitlines()
    frames = [l.strip() for l in lines if l.strip()]
    return frames


def _find_lidar_file(root_path, token):
    # common pattern: annotations/velodyne_points/<token>.bin
    cand = osp.join(root_path, 'annotations', 'velodyne_points', f'{token}.bin')
    if osp.exists(cand):
        return cand
    # try numeric index variants
    cand2 = osp.join(root_path, 'annotations', 'velodyne_points', f'{token}.pcd')
    if osp.exists(cand2):
        return cand2
    return osp.join(root_path, 'velodyne_points', f'{token}.bin')


def _raw_img_path(raw_root, folder, token):
    for ext in ('.png', '.jpg', '.jpeg'):
        p = osp.join(raw_root, 'data', folder, f'{token}{ext}')
        if osp.exists(p):
            return p
    return ''


def _ensure_placeholder_image(path, size=(1600, 900), color=(0, 0, 0)):
    mmcv.mkdir_or_exist(osp.dirname(path))
    if not osp.exists(path):
        Image.new('RGB', size, color=color).save(path)


def _ensure_placeholder_depth(path):
    mmcv.mkdir_or_exist(osp.dirname(path))
    if not osp.exists(path):
        # Nx3 array, the loader expects cam_depth[:, :2] and cam_depth[:, 2].
        np.save(path, np.array([[0.0, 0.0, 1.0]], dtype=np.float32))


def _build_cam_map(root_path, raw_root, token):
    # Map available camera folders to nuScenes camera keys.
    data_dir = osp.join(raw_root, 'data')
    cams = {}
    key_to_folder = {}
    for folder, key in RAW_CAM_FOLDERS.items():
        if osp.isdir(osp.join(data_dir, folder)):
            key_to_folder[key] = folder

    available_folders = list(key_to_folder.values())
    fallback_folder = available_folders[0] if available_folders else None

    for key in CAM_KEY_ORDER:
        folder = key_to_folder.get(key, fallback_folder)
        if folder is None:
            cams[key] = dict(
                data_path='',
                sample_data_token=str(token),
                cam_intrinsic=np.eye(3, dtype=np.float32),
                sensor2lidar_rotation=np.eye(3, dtype=np.float32),
                sensor2lidar_translation=np.zeros(3, dtype=np.float32),
            )
            continue

        raw_img = _raw_img_path(raw_root, folder, token)
        if raw_img:
            rel_img = osp.join('data', 'bevlog', 'data', folder, f'{token}.png')
        else:
            rel_img = osp.join('data', 'bevlog', 'projmap', key, f'{token}.jpg')

        cams[key] = dict(
            data_path=rel_img,
            sample_data_token=str(token),
            cam_intrinsic=np.array([[1200.0, 0.0, 800.0],
                                   [0.0, 1200.0, 450.0],
                                   [0.0, 0.0, 1.0]], dtype=np.float32),
            sensor2lidar_rotation=np.eye(3, dtype=np.float32),
            sensor2lidar_translation=np.zeros(3, dtype=np.float32),
        )

    # make sure the auxiliary projection/depth/bev artifacts exist
    for cam in CAM_KEY_ORDER:
        cam_dir = osp.join(root_path, 'projmap', cam)
        depth_dir = osp.join(root_path, 'projdepth', cam)
        mmcv.mkdir_or_exist(cam_dir)
        mmcv.mkdir_or_exist(depth_dir)
        proj_path = osp.join(cam_dir, f'{token}.jpg')
        depth_path = osp.join(depth_dir, f'{token}.npy')
        src_folder = key_to_folder.get(cam, fallback_folder)
        src_img = _raw_img_path(raw_root, src_folder, token) if src_folder is not None else ''
        if src_img and not osp.exists(proj_path):
            Image.open(src_img).convert('RGB').save(proj_path)
        _ensure_placeholder_image(proj_path)
        _ensure_placeholder_depth(depth_path)

    bev_dir = osp.join(root_path, 'bevmap')
    mmcv.mkdir_or_exist(bev_dir)
    bev_path = osp.join(bev_dir, f'{token}.png')
    if not osp.exists(bev_path):
        _ensure_placeholder_image(bev_path, size=(512, 512), color=(255, 255, 255))

    return cams


def create_bevlog_infos(root_path, info_prefix='nuscenes', version=None, max_sweeps=0):
    """Create minimal infos for BEVLOG recordings.

    Args:
        root_path (str): dataset root (e.g. data/bevlog)
        info_prefix (str): prefix for saved files (keep `nuscenes` for compatibility)
    """
    root_path = osp.abspath(root_path)
    raw_root = root_path + '_raw'
    frames = _collect_frame_list(root_path)
    infos = []
    for idx, token in enumerate(frames):
        lidar_path = _find_lidar_file(root_path, token)
        cams = _build_cam_map(root_path, raw_root, token)
        empty_boxes = np.zeros((0, 7), dtype=np.float32)
        empty_labels = np.zeros((0,), dtype=np.int64)
        empty_valid = np.zeros((0,), dtype=np.bool_)
        empty_velocity = np.zeros((0, 2), dtype=np.float32)
        info = {
            'token': str(token),
            'timestamp': int(float(token) * 1e6) if str(token).replace('.', '', 1).isdigit() else idx,
            'lidar_path': lidar_path,
            'sweeps': [],
            'cams': cams,
            'lidar2ego_rotation': [1.0, 0.0, 0.0, 0.0],
            'lidar2ego_translation': [0.0, 0.0, 0.0],
            'ego2global_rotation': [1.0, 0.0, 0.0, 0.0],
            'ego2global_translation': [0.0, 0.0, 0.0],
            'gt_boxes': empty_boxes,
            'gt_names': np.array([], dtype='<U1'),
            'num_lidar_pts': empty_labels,
            'valid_flag': empty_valid,
            'gt_velocity': empty_velocity,
        }
        # ensure placeholder lidar file exists for the smoke test
        mmcv.mkdir_or_exist(osp.dirname(lidar_path))
        if not osp.exists(lidar_path):
            open(lidar_path, 'wb').close()
        infos.append(info)

    # split train/val (90/10)
    n = len(infos)
    split = int(n * 0.9)
    train_infos = infos[:split]
    val_infos = infos[split:]

    metadata = dict(version='bevlog')
    mmcv.mkdir_or_exist(root_path)
    mmcv.dump(dict(infos=train_infos, metadata=metadata),
              osp.join(root_path, f'{info_prefix}_infos_train.pkl'))
    mmcv.dump(dict(infos=val_infos, metadata=metadata),
              osp.join(root_path, f'{info_prefix}_infos_val.pkl'))
    print(f'Generated {len(train_infos)} train and {len(val_infos)} val samples at {root_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert BEVLOG dataset to nuscenes-style info pkls')
    parser.add_argument('--root-path', type=str, required=True)
    parser.add_argument('--info-prefix', type=str, default='nuscenes')
    args = parser.parse_args()
    create_bevlog_infos(args.root_path, info_prefix=args.info_prefix)
