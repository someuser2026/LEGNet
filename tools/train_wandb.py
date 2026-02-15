# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
import torch.distributed as dist
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash
from mmdet import __version__
from mmdet.apis import init_random_seed, set_random_seed

from mmrotate.apis import train_detector
from mmrotate.datasets import build_dataset
from mmrotate.models import build_detector
from mmrotate.utils import collect_env, get_root_logger, setup_multi_processes


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a detector with W&B name/imgsz helpers')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--auto-resume',
        action='store_true',
        help='resume from the latest checkpoint automatically')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    parser.add_argument(
        '--wandb-project',
        default=None,
        help='override W&B project name (defaults to config/project setting)')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--diff-seed',
        action='store_true',
        help='Whether or not set different seeds for different ranks')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def _extract_imgsz(cfg):
    train_cfg = cfg.get('data', {}).get('train', {})
    pipeline = train_cfg.get('pipeline', [])
    for step in pipeline:
        if not isinstance(step, dict):
            continue
        if step.get('type') in ('RResize', 'Resize'):
            imgsz = step.get('img_scale', step.get('scale'))
            if isinstance(imgsz, list) and len(imgsz) == 1:
                return imgsz[0]
            return imgsz
    return None


def _resolve_wandb_project(cfg, arg_project):
    if arg_project:
        return arg_project
    log_cfg = cfg.get('log_config', {})
    for hook in log_cfg.get('hooks', []):
        if hook.get('type') == 'WandbLoggerHook':
            init_kwargs = hook.get('init_kwargs', {})
            if init_kwargs.get('project'):
                return init_kwargs['project']
    return 'legnet-obb'


def _patch_wandb_runtime(cfg, config_path, arg_project):
    run_name = osp.splitext(osp.basename(config_path))[0]
    imgsz = _extract_imgsz(cfg)
    project = _resolve_wandb_project(cfg, arg_project)

    old_log_cfg = cfg.get('log_config', {})
    interval = old_log_cfg.get('interval', 50)
    old_hooks = old_log_cfg.get('hooks', [])

    # Keep non-text/non-wandb hooks.
    passthrough_hooks = []
    has_text_hook = False
    wandb_extra = {}
    old_init_kwargs = {}
    for hook in old_hooks:
        hook_type = hook.get('type')
        if hook_type == 'TextLoggerHook':
            has_text_hook = True
            continue
        if hook_type == 'WandbLoggerHook':
            wandb_extra = {
                k: v
                for k, v in hook.items() if k not in ('type', 'init_kwargs')
            }
            old_init_kwargs = hook.get('init_kwargs', {})
            continue
        passthrough_hooks.append(hook)

    hooks = []
    if has_text_hook:
        hooks.append(dict(type='TextLoggerHook'))
    else:
        hooks.append(dict(type='TextLoggerHook'))
    hooks.extend(passthrough_hooks)

    wandb_init_kwargs = dict(old_init_kwargs)
    wandb_init_kwargs.update(dict(project=project, name=run_name))
    wandb_hook = dict(type='WandbLoggerHook', init_kwargs=wandb_init_kwargs)
    wandb_hook.update(wandb_extra)
    hooks.append(wandb_hook)
    cfg.log_config = dict(interval=interval, hooks=hooks)

    # Register custom hook module import.
    hook_module = 'mmrotate.utils.wandb_imgsz_hook'
    if cfg.get('custom_imports', None) is None:
        cfg.custom_imports = dict(
            imports=[hook_module], allow_failed_imports=False)
    else:
        imports = list(cfg.custom_imports.get('imports', []))
        if hook_module not in imports:
            imports.append(hook_module)
        cfg.custom_imports['imports'] = imports
        if 'allow_failed_imports' not in cfg.custom_imports:
            cfg.custom_imports['allow_failed_imports'] = False

    custom_hooks = list(cfg.get('custom_hooks', []))
    custom_hooks = [
        hook for hook in custom_hooks if hook.get('type') != 'WandbImgszHook'
    ]
    custom_hooks.append(dict(type='WandbImgszHook', imgsz=imgsz, priority='LOW'))
    cfg.custom_hooks = custom_hooks


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    _patch_wandb_runtime(cfg, args.config, args.wandb_project)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.auto_resume = args.auto_resume
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
        if len(cfg.gpu_ids) > 1:
            warnings.warn(
                f'We treat {cfg.gpu_ids} as gpu-ids, and reset to '
                f'{cfg.gpu_ids[0:1]} as gpu-ids to avoid potential error in '
                'non-distribute training time.')
            cfg.gpu_ids = cfg.gpu_ids[0:1]
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    meta = dict()
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    seed = init_random_seed(args.seed)
    seed = seed + dist.get_rank() if args.diff_seed else seed
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta['seed'] = seed
    meta['exp_name'] = osp.basename(args.config)

    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__ + get_git_hash()[:7],
            CLASSES=datasets[0].CLASSES)
    model.CLASSES = datasets[0].CLASSES
    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
