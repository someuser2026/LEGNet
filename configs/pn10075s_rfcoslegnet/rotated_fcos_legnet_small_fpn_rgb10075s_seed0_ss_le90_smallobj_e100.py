_base_ = './rotated_fcos_legnet_small_fpn_rgb10075s_seed0_ss_le90_e36.py'

# Tuned for small objects in the ~5x5 to 100x100 px range.
model = dict(
    neck=dict(start_level=0),
    bbox_head=dict(
        strides=[4, 8, 16, 32, 64],
        regress_ranges=((-1, 32), (32, 64), (64, 128), (128, 256),
                        (256, 100000000.0))))

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[36, 67, 92])

runner = dict(type='EpochBasedRunner', max_epochs=100)
checkpoint_config = dict(interval=1, max_keep_ckpts=1)
