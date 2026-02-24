#!/usr/bin/env python3
"""Convert MMRotate .pkl detections to YOLO polygon prediction txt files.

Output line format:
    cls_id x1 y1 x2 y2 x3 y3 x4 y4 conf
"""

import argparse
import os
import os.path as osp
import pickle

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert MMRotate .pkl predictions to YOLO polygon txt.')
    parser.add_argument('config', help='Config used for testing.')
    parser.add_argument('pkl', help='Path to .pkl predictions from tools/test.py --out.')
    parser.add_argument('out_dir', help='Directory to write YOLO txt files.')
    parser.add_argument(
        '--split',
        default='test',
        choices=['train', 'val', 'test'],
        help='Which dataset split in config to align with prediction order.')
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.0,
        help='Discard predictions with confidence below this threshold.')
    parser.add_argument(
        '--version',
        default=None,
        choices=['oc', 'le90', 'le135'],
        help='Angle version override. Defaults to dataset.version.')
    parser.add_argument(
        '--img-prefix',
        default=None,
        help='Optional override for dataset image directory.')
    parser.add_argument(
        '--absolute-coords',
        action='store_true',
        help='Write absolute pixel polygon coords. Default writes normalized [0,1] coords.')
    parser.add_argument(
        '--save-class-names',
        action='store_true',
        help='Also write class names into out_dir/classes.txt.')
    return parser.parse_args()


def _build_dataset(cfg, split):
    from mmrotate.datasets import build_dataset

    data_cfg = cfg.data[split]
    if isinstance(data_cfg, list):
        raise ValueError(
            f'Unsupported: cfg.data.{split} is a list. Use a single dataset config.')
    data_cfg.test_mode = True
    return build_dataset(data_cfg)


def _load_results(pkl_path):
    with open(pkl_path, 'rb') as f:
        results = pickle.load(f)
    if not isinstance(results, list):
        raise TypeError(f'Expected list from {pkl_path}, got {type(results)}')
    if len(results) == 0:
        return results
    if isinstance(results[0], tuple):
        results = [item[0] for item in results]
    return results


def _resolve_img_path(filename, img_prefix, override_prefix):
    if osp.isabs(filename):
        return filename
    if override_prefix is not None:
        return osp.join(override_prefix, filename)
    if img_prefix is None:
        return filename
    return osp.join(img_prefix, filename)


def _to_poly_and_score(dets, version):
    from mmrotate.core import obb2poly_np

    if dets.size == 0:
        return np.zeros((0, 8), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    if dets.ndim == 1:
        dets = dets[None, :]

    if dets.shape[1] in (5, 6):
        if dets.shape[1] == 5:
            ones = np.ones((dets.shape[0], 1), dtype=dets.dtype)
            dets = np.concatenate([dets, ones], axis=1)
        else:
            dets = dets[:, :6]
        polys = obb2poly_np(dets, version)
        return polys[:, :8], polys[:, 8]

    if dets.shape[1] in (8, 9):
        polys = dets[:, :8]
        if dets.shape[1] == 9:
            scores = dets[:, 8]
        else:
            scores = np.ones((dets.shape[0],), dtype=dets.dtype)
        return polys, scores

    raise ValueError(
        f'Unsupported detection shape {dets.shape}. Expected Nx5/6 OBB or Nx8/9 polygon.')


def _normalize_polys(polys, width, height):
    polys = polys.copy()
    polys[:, 0::2] = np.clip(polys[:, 0::2] / float(width), 0.0, 1.0)
    polys[:, 1::2] = np.clip(polys[:, 1::2] / float(height), 0.0, 1.0)
    return polys


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    try:
        import cv2
        from mmcv import Config
    except ImportError as e:
        raise ImportError(
            'Missing dependency. Please run this script in your MMRotate env '
            '(requires mmcv + mmrotate + opencv-python).') from e

    cfg = Config.fromfile(args.config)
    dataset = _build_dataset(cfg, args.split)
    results = _load_results(args.pkl)
    if len(results) != len(dataset):
        raise ValueError(
            f'Prediction count ({len(results)}) != dataset size ({len(dataset)}).')

    version = args.version or getattr(dataset, 'version', 'oc')
    img_prefix = getattr(dataset, 'img_prefix', None)

    for idx, result in enumerate(results):
        data_info = dataset.data_infos[idx]
        filename = data_info['filename']
        txt_name = osp.splitext(osp.basename(filename))[0] + '.txt'
        out_path = osp.join(args.out_dir, txt_name)

        if not isinstance(result, list):
            raise TypeError(
                f'Per-image result at idx={idx} must be list, got {type(result)}')

        img_path = _resolve_img_path(filename, img_prefix, args.img_prefix)
        if args.absolute_coords:
            img_w, img_h = None, None
        else:
            if not osp.exists(img_path):
                raise FileNotFoundError(
                    f'Image not found for normalization: {img_path}')
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f'Unable to read image: {img_path}')
            img_h, img_w = img.shape[:2]

        lines = []
        for cls_id, dets in enumerate(result):
            dets = np.asarray(dets)
            polys, scores = _to_poly_and_score(dets, version)
            if polys.shape[0] == 0:
                continue

            keep = scores >= args.score_thr
            if not np.any(keep):
                continue

            polys = polys[keep]
            scores = scores[keep]

            if not args.absolute_coords:
                polys = _normalize_polys(polys, img_w, img_h)

            for poly, score in zip(polys, scores):
                row = [str(cls_id)]
                row.extend(f'{float(v):.6f}' for v in poly.tolist())
                row.append(f'{float(score):.6f}')
                lines.append(' '.join(row))

        with open(out_path, 'w') as f:
            if lines:
                f.write('\n'.join(lines) + '\n')

    if args.save_class_names and hasattr(dataset, 'CLASSES'):
        cls_path = osp.join(args.out_dir, 'classes.txt')
        with open(cls_path, 'w') as f:
            for name in dataset.CLASSES:
                f.write(f'{name}\n')

    print(f'Wrote YOLO polygon predictions to: {args.out_dir}')


if __name__ == '__main__':
    main()
