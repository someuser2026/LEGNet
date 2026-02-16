# import os
# import os.path as osp


# def _wandb_run_name():
#     # Preferred source: explicit env from launcher/wrapper.
#     cfg_path = os.environ.get('LEGNET_CONFIG_PATH', '')
#     if cfg_path:
#         return osp.splitext(osp.basename(cfg_path))[0]

#     env_name = os.environ.get('WANDB_RUN_NAME', '')
#     if env_name:
#         return env_name

#     return 'test_run_debug'


# _wandb_project = os.environ.get('WANDB_PROJECT', 'legnet-obb')

# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(project="mmrotate", name=None))
    ])
# yapf:enable

# Make custom hook available when this runtime is used.
custom_imports = dict(
    imports=['mmrotate.utils.wandb_imgsz_hook'],
    allow_failed_imports=False)
custom_hooks = [
    # `imgsz` is auto-filled by tools/train_wandb.py when available.
    dict(type='WandbImgszHook', imgsz=None, priority='LOW')
]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1), ('val', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'
