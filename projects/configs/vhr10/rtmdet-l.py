_base_ = [
    'mmdet::_base_/default_runtime.py',
    'mmdet::_base_/schedules/schedule_2x.py',
    'mmdet::_base_/datasets/coco_detection.py'
]

load_from = '/workspace/mm_sam/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth'
# compile=True

model = dict(
    type='RTMDet',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        bgr_to_rgb=False,
        batch_augments=None),
    backbone=dict(
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=1,
        widen_factor=1,
        channel_attention=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU', inplace=True)),
    neck=dict(
        type='CSPNeXtPAFPN',
        in_channels=[256, 512, 1024],
        out_channels=256,
        num_csp_blocks=3,
        expand_ratio=0.5,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU', inplace=True)),
    bbox_head=dict(
        type='RTMDetSepBNHead',
        num_classes=10,
        in_channels=256,
        stacked_convs=2,
        feat_channels=256,
        anchor_generator=dict(
            type='MlvlPointGenerator', offset=0, strides=[8, 16, 32]),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        with_objectness=False,
        exp_on_reg=True,
        share_conv=True,
        pred_kernel_size=1,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU', inplace=True)),
    train_cfg=dict(
        assigner=dict(type='DynamicSoftLabelAssigner', topk=13),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=30000,
        min_bbox_size=0,
        score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.65),
        max_per_img=300),
)

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomResize',
        scale=(1024, 1024),
        ratio_range=(0.1, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(1024, 1024)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(1024, 1024), pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='Pad', size=(1024, 1024), pad_val=dict(img=(114, 114, 114))),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

data_root = '/workspace/datasets/vhr10/'
meta_info = dict(
    classes=('airplane', 'ship', 'storage tank', 'baseball diamond', 'tennis court',
             'basketball court', 'ground track field', 'harbor', 'bridge', 'vehicle'),
    palette=[[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [
        0, 255, 255], [255, 255, 255], [128, 0, 0], [0, 128, 0], [0, 0, 128]]
)


train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    batch_sampler=None,
    pin_memory=True,
    dataset=dict(
        data_root=data_root,
        metainfo=meta_info,
        ann_file='instances_train2017.json',
        data_prefix=dict(img='positive image set/'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        data_root=data_root,
        metainfo=meta_info,
        ann_file='instances_val2017.json',
        data_prefix=dict(img='positive image set/'),
        pipeline=test_pipeline))
test_dataloader=val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'instances_val2017.json',
    metric='bbox',
    format_only=False)
test_evaluator = val_evaluator

max_epochs=24
base_lr=0.004
interval=12

train_cfg=dict(
    max_epochs=max_epochs,
    val_interval=interval)

# optimizer
optim_wrapper=dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

# learning rate
param_scheduler=[
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        # use cosine lr from 150 to 300 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

# hooks
default_hooks=dict(
    checkpoint=dict(
        interval=interval,
        max_keep_ckpts=3  # only keep latest 3 checkpoints
    ))
custom_hooks=[
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49)
]
