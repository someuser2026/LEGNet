_base_ = './rotated_fcos_legnet_small_fpn_rgb10075s_seed0_ss_le90_e36.py'

custom_imports = dict(
    imports=['mmrotate.models.dense_heads.kfiou_rotated_fcos_head'],
    allow_failed_imports=False)

model = dict(
    bbox_head=dict(
        type='KFIoURFCOSHead',
        loss_bbox=dict(
            _delete_=True,
            type='KFLoss',
            fun='ln',
            loss_weight=1.0)))
