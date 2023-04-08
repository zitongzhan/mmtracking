_base_ = ['./youtube_vis.py']

dataset_type = 'OVISDataset'
data_root = 'data/ovis2021/'
dataset_version = 'ovis2021'

train_pipeline = [
    dict(
        type='TransformBroadcaster',
        share_random_params=True,
        transforms=[
            dict(type='LoadImageFromFile'),
            dict(
                type='LoadTrackAnnotations',
                with_instance_id=True,
                with_mask=True,
                with_bbox=True),
            dict(type='mmdet.Resize', scale=(960, 480), keep_ratio=True),
            dict(type='mmdet.RandomFlip', prob=0.5),
        ]),
    dict(type='PackTrackInputs', ref_prefix='ref', num_key_frames=1)
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadTrackAnnotations',
        with_instance_id=True,
        with_mask=True,
        with_bbox=True),
    dict(type='mmdet.Resize', scale=(960, 480), keep_ratio=True),
    dict(type='PackTrackInputs', pack_single_img=True)
]

# dataloader
train_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        dataset_version=dataset_version,
        ann_file='coco_vid_ann/youtube_vis_2019_train.json',
        data_prefix=dict(img_path='train/'),
        pipeline=train_pipeline,
    ))
val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        dataset_version=dataset_version,
        ann_file='coco_vid_ann/youtube_vis_2019_valid.json',
        data_prefix=dict(img_path='valid/'),
        pipeline=test_pipeline,
    ))
test_dataloader = val_dataloader
