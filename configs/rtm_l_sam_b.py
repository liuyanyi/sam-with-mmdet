from functools import partial
import torch

_base_ = 'mmdet::_base_/default_runtime.py'

custom_imports = dict(
    imports=['projects.sam'], allow_failed_imports=False)

prompt_embed_dim = 256
image_size = 1024
vit_patch_size = 16
image_embedding_size = image_size // vit_patch_size

model = dict(
    type='SAM',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=False,
        pad_size_divisor=32),
    backbone=dict(
        type='ImageEncoderViT',
        depth=12,
        embed_dim=768,
        img_size=image_size,
        mlp_ratio=4,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        num_heads=12,
        patch_size=16,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=[2, 5, 8, 11],
        window_size=14,
        out_chans=prompt_embed_dim),
    neck=None,
    prompt_generator=dict(
        type='RTMDet',
        data_preprocessor=dict(
            type='DetDataPreprocessor',
            mean=[103.53, 116.28, 123.675],
            std=[57.375, 57.12, 58.395],
            bgr_to_rgb=True,
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
            num_classes=80,
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
    ),
    prompt_encoder=dict(
        type='PromptEncoder',
        embed_dim=prompt_embed_dim,
        image_embedding_size=(image_embedding_size, image_embedding_size),
        input_image_size=(image_size, image_size),
        mask_in_chans=16),
    mask_decoder=dict(
        type='MaskDecoder',
        num_multimask_outputs=3,
        transformer=dict(
            type='TwoWayTransformer',
            depth=2,
            embedding_dim=prompt_embed_dim,
            mlp_dim=2048,
            num_heads=8,
        ),
        transformer_dim=prompt_embed_dim,
        iou_head_depth=3,
        iou_head_hidden_dim=256),
    train_cfg=None,
    test_cfg=dict(
        points_per_side=32,
        multimask_output=False,
        points_per_batch=64,
        pred_iou_thresh=0.7,
        stability_score_thresh=0,
        stability_score_offset=1.0,
        nms=dict(type='nms', iou_threshold=0.5))
)

dataset_type = 'CocoDataset'
data_root = 'data/coco/'

# Example to use different file client
# Method 1: simply set the data root and let the file I/O module
# automatically infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/detection/coco/'

# Method 2: Use `backend_args`, `file_client_args` in versions before 3.0.0rc6
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection/',
#         'data/': 's3://openmmlab/datasets/detection/'
#     }))
backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='Pad', size=(1024, 1024), pad_val=dict(img=(114, 114, 114))),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader
# visualizer = dict(type='MaskVisualizer', alpha=0.4)