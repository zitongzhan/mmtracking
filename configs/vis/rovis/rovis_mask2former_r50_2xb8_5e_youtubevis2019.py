_base_ = [
    './mask2former_iou_r50_4xb4-5e_youtubevis2019.py',
]

model = dict(
    type='ROVIS',
    detector=dict(
        panoptic_head=dict(type='mmtrack.AdditionalQueriesFormerHead', ), ),
    tracker=dict(
        _delete_=True,
        type='TrackFormerTracker',
        # match_weights=dict(det_score=1.0, iou=2.0, det_label=10.0),
        # num_frames_retain=20
        nms_cfg=dict(
            nms_pre=500,
            filter_thr=0.05,
            kernel='linear',  # gaussian/linear
            sigma=2.0,
            max_per_img=100)),
    # model training and testing settings
    train_cfg=dict(
        # https://github.com/timmeinhardt/trackformer/blob/df70fef0539dc6ebe8ed26bf1ce55dd6e8f87968/cfgs/train.yaml#L58
        track_query_false_positive_prob=0.2,  # the probability of 'adding'
        track_query_false_negative_prob=0.8,  # the probability of 'keeping'
    ),
)
