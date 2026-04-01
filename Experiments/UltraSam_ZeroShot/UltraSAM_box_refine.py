N_POINTS = 1
backend_args = None
custom_hooks = [
    dict(type='MonkeyPatchHook'),
]
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'mmpretrain.models',
        'endosam.datasets.transforms.custom_pipeline',
        'endosam.datasets.transforms.point_formatting',
        'endosam.visualization.point_visualization',
        'endosam.models.detectors.SAM',
        'endosam.models.backbones.vit_sam',
        'endosam.models.backbones.MED_SA',
        'endosam.models.dense_heads.sam_mask_decoder',
        'endosam.models.dense_heads.sam_mask_class_decoder',
        'endosam.datasets.evaluation.LabelMetric',
        'endosam.models.utils.sam_layers',
        'endosam.models.task_modules.assigners.SAMassigner',
        'endosam.models.task_modules.prior_generators.prompt_encoder',
        'endosam.models.task_modules.prior_generators.label_encoder',
        'endosam.hooks.MonkeyPatchHook',
        'endosam.hooks.FreezeHook',
        'endosam.hooks.ValLossHook',
    ])
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    pad_size_divisor=1024,
    std=[
        58.395,
        57.12,
        57.375,
    ],
    type='DetDataPreprocessor')
data_root = 'UltraSAM_DATA/UltraSAM'
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False,
        interval=2000,
        max_keep_ckpts=1,
        save_best='coco/segm_mAP',
        save_optimizer=False,
        type='CheckpointHook'),
    logger=dict(interval=50, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(draw=False, type='DetVisualizationHook'))
default_scope = 'mmdet'
dummy_metainfo = dict(classes=('object', ))
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
file_client_args = dict(backend='disk')
launcher = 'none'
load_from = '/home/maia-user/yufeng/UltraSam/UltraSam.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=False, type='LogProcessor', window_size=50)
model = dict(
    backbone=dict(
        arch='base',
        img_size=1024,
        out_channels=256,
        patch_size=16,
        type='mmpretrain.ViTSAM',
        use_abs_pos=True,
        use_rel_pos=True,
        window_size=14),
    bbox_head=dict(type='SAMHead'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_size_divisor=1024,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='DetDataPreprocessor'),
    decoder=dict(
        layer_cfg=dict(
            embedding_dim=256,
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.1),
            num_heads=8),
        num_layers=2),
    prompt_encoder=dict(
        label_encoder=dict(embed_dims=256, type='LabelEmbedEncoder'),
        type='SAMPaddingGenerator'),
    train_cfg=dict(assigner=dict(type='SAMassigner')),
    type='SAM',
    use_mask_refinement=True)
optim_wrapper = dict(
    clip_grad=dict(max_norm=0.1, norm_type=2),
    optimizer=dict(lr=0.0001, type='AdamW', weight_decay=0.0001),
    type='OptimWrapper')
orig_test_evaluator = dict(
    ann_file=
    'UltraSAM_DATA/UltraSAM/MMOTU_2d/annotations/test.agnostic.MMOTU_2d__coco.json',
    backend_args=None,
    classwise=True,
    format_only=False,
    metric=[
        'bbox',
        'segm',
    ],
    type='CocoMetric')
orig_val_evaluator = dict(
    ann_file='UltraSAM_DATA/UltraSAM/test.agnostic.noSmall.coco.json',
    backend_args=None,
    classwise=True,
    format_only=False,
    metric=[
        'bbox',
        'segm',
    ],
    type='CocoMetric')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=500, start_factor=0.001, type='LinearLR'),
    dict(
        begin=0,
        by_epoch=False,
        end=30000,
        gamma=0.1,
        milestones=[
            20000,
            28888,
        ],
        type='MultiStepLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='test.coco.json',
        backend_args=None,
        data_prefix=dict(img=''),
        data_root='/home/maia-user/yufeng/Data/BUS_Mix_Seg/',
        metainfo=dict(classes=('object', )),
        pipeline=[
            dict(
                file_client_args=dict(backend='disk'),
                type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1024,
                1024,
            ), type='FixScaleResize'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(
                get_center_point=True,
                normalize=False,
                number_of_points=[
                    1,
                ],
                test=True,
                type='GetPointFromBox'),
            dict(
                max_jitter=0.0, normalize=False, test=True,
                type='GetPointBox'),
            dict(prompt_probabilities=[
                0.0,
                1.0,
            ], type='GetPromptType'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackPointDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='/home/maia-user/yufeng/Data/BUS_Mix_Seg/test.coco.json',
    backend_args=None,
    classwise=True,
    format_only=False,
    metric=[
        'bbox',
        'segm',
    ],
    type='CocoMetric')
test_pipeline = [
    dict(file_client_args=dict(backend='disk'), type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        1024,
        1024,
    ), type='FixScaleResize'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        get_center_point=True,
        normalize=False,
        number_of_points=[
            1,
        ],
        test=True,
        type='GetPointFromBox'),
    dict(max_jitter=0.0, normalize=False, test=True, type='GetPointBox'),
    dict(prompt_probabilities=[
        0.0,
        1.0,
    ], type='GetPromptType'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackPointDetInputs'),
]
train_cfg = dict(max_iters=30000, type='IterBasedTrainLoop', val_interval=5000)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=8,
    dataset=dict(
        ann_file='train.agnostic.noSmall.coco.json',
        backend_args=None,
        data_prefix=dict(img=''),
        data_root='UltraSAM_DATA/UltraSAM',
        filter_cfg=dict(filter_empty_gt=True),
        metainfo=dict(classes=('object', )),
        pipeline=[
            dict(
                file_client_args=dict(backend='disk'),
                type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(prob=0.5, type='RandomFlip'),
            dict(keep_ratio=True, scale=(
                1024,
                1024,
            ), type='FixScaleResize'),
            dict(
                normalize=False,
                number_of_points=[
                    1,
                ],
                test=False,
                type='GetPointFromMask'),
            dict(normalize=False, test=False, type='GetPointBox'),
            dict(prompt_probabilities=[
                0.5,
                0.5,
            ], type='GetPromptType'),
            dict(type='PackPointDetInputs'),
        ],
        type='CocoDataset'),
    num_workers=6,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_pipeline = [
    dict(file_client_args=dict(backend='disk'), type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(prob=0.5, type='RandomFlip'),
    dict(keep_ratio=True, scale=(
        1024,
        1024,
    ), type='FixScaleResize'),
    dict(
        normalize=False,
        number_of_points=[
            1,
        ],
        test=False,
        type='GetPointFromMask'),
    dict(normalize=False, test=False, type='GetPointBox'),
    dict(prompt_probabilities=[
        0.5,
        0.5,
    ], type='GetPromptType'),
    dict(type='PackPointDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='val.agnostic.noSmall.coco.json',
        backend_args=None,
        data_prefix=dict(img=''),
        data_root='UltraSAM_DATA/UltraSAM',
        metainfo=dict(classes=('object', )),
        pipeline=[
            dict(
                file_client_args=dict(backend='disk'),
                type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1024,
                1024,
            ), type='FixScaleResize'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(
                get_center_point=True,
                normalize=False,
                number_of_points=[
                    1,
                ],
                test=True,
                type='GetPointFromBox'),
            dict(
                max_jitter=0.0, normalize=False, test=True,
                type='GetPointBox'),
            dict(prompt_probabilities=[
                0.0,
                1.0,
            ], type='GetPromptType'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackPointDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='UltraSAM_DATA/UltraSAM/test.agnostic.noSmall.coco.json',
    backend_args=None,
    classwise=True,
    format_only=False,
    metric=[
        'bbox',
        'segm',
    ],
    type='CocoMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='PointVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ])
work_dir = '/home/maia-user/yufeng/BUS-Lesion-Seg/Experiments/UltraSam_ZeroShot'
