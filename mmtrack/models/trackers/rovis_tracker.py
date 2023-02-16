# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmdet.models.layers.matrix_nms import mask_matrix_nms
from mmengine.structures import InstanceData

from mmtrack.registry import MODELS
from .base_tracker import BaseTracker


@MODELS.register_module()
class TrackFormerTracker(BaseTracker):
    """Tracker for .

    Args:
        match_weights (dict[str : float]): The Weighting factor when computing
        the match score. It contains keys as follows:

            - det_score (float): The coefficient of `det_score` when computing
                match score.
            - iou (float): The coefficient of `ious` when computing match
                score.
            - det_label (float): The coefficient of `label_deltas` when
                computing match score.

        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(
            self,
            #  match_weights=dict(det_score=1.0, iou=2.0, det_label=10.0),
            nms_cfg=None,
            **kwargs):
        super().__init__(**kwargs)
        # self.match_weights = match_weights
        # need a new inactive track dict
        self.nms_cfg = nms_cfg
        self.idol_nms = True

    def track(self,
              img,
              img_metas,
              model,
              data_sample,
              frame_id,
              rescale=False,
              **kwargs):
        """Tracking forward function.

        Args:
            img (Tensor): of shape (1, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            model (nn.Module): VIS model.
            feats (N, C): transformer decoder features of the input image.
            query_ids (torch.Tensor): of shape (N, )
            bboxes (Tensor): of shape (N, 5). N is the number of queries.
            labels (Tensor): of shape (N, ).
            masks (Tensor): of shape (N, H, W)
            frame_id (int): The id of current frame, 0-index.
            rescale (bool, optional): If True, the bounding boxes should be
                rescaled to fit the original scale of the image. Defaults to
                False.

        Returns:
            tuple: Tracking results.
        """
        pred_det_instances = data_sample.pred_det_instances

        if pred_det_instances.bboxes.shape[0] == 0:
            ids = torch.zeros_like(pred_det_instances.labels)
            pred_det_instances.instances_id = ids
            return pred_det_instances

        rescaled_bboxes = pred_det_instances.bboxes.clone()
        if rescale:
            rescaled_bboxes[:, :4] *= torch.tensor(
                img_metas[0]['scale_factor']).to(
                    pred_det_instances.bboxes.device)
            pred_det_instances.bboxes = rescaled_bboxes

        if self.empty:
            prev_ids = torch.tensor([], dtype=torch.int64)
            prev_track_static_query_ids = torch.tensor([], dtype=torch.int64)
        else:
            prev_ids = self.get('ids')
            prev_input_queries = self.get('feats')
            prev_track_static_query_ids = self.get('track_static_query_ids')

        # first handle the existing tracks
        num_fixed_query = model.detector.panoptic_head.num_queries
        existing_tracks_mask = pred_det_instances.query_ids >= num_fixed_query

        # perform non-maxima suppression on the tracks
        if existing_tracks_mask.sum() > 0:
            query_ids = pred_det_instances.query_ids
            existing_tracks = pred_det_instances[existing_tracks_mask]

            existing_tracks_ids = prev_ids[query_ids[existing_tracks_mask] -
                                           num_fixed_query]
            existing_tracks_input_queries = prev_input_queries[
                query_ids[existing_tracks_mask] - num_fixed_query]
            existing_tracks_static_query_ids = prev_track_static_query_ids[
                query_ids[existing_tracks_mask] - num_fixed_query]

            existing_tracks.ids = existing_tracks_ids

            if self.idol_nms:
                keep_inds = mask_nms(
                    existing_tracks.masks,
                    existing_tracks.scores,
                    # labels=existing_tracks_labels,
                    nms_thr=0.7)
                existing_tracks = existing_tracks[keep_inds]

            else:
                (existing_tracks_scores, existing_tracks_labels,
                 existing_tracks_masks, keep_inds) = mask_matrix_nms(
                     existing_tracks.masks,
                     existing_tracks.labels,
                     existing_tracks.scores.to(existing_tracks.masks.device),
                     nms_pre=self.nms_cfg.nms_pre,
                     max_num=self.nms_cfg.max_per_img,
                     kernel=self.nms_cfg.kernel,
                     sigma=self.nms_cfg.sigma,
                     filter_thr=0.1)
                keep_inds = keep_inds.to('cpu')
                existing_tracks_scores = existing_tracks_scores.to('cpu')
                existing_tracks = existing_tracks[keep_inds]
                existing_tracks.scores = existing_tracks_scores
                existing_tracks.labels = existing_tracks_labels
                existing_tracks.masks = existing_tracks_masks

            existing_tracks_input_queries = existing_tracks_input_queries[
                keep_inds]
            existing_tracks_static_query_ids = \
                existing_tracks_static_query_ids[keep_inds]

            # discard low quality object queries
            query_update_threshold = 0.3
            existing_tracks.feats[
                existing_tracks.scores <
                query_update_threshold] = existing_tracks_input_queries[
                    existing_tracks.scores < query_update_threshold]

        # fixed queries generate new tracks
        new_tracks_mask = pred_det_instances.query_ids < num_fixed_query
        new_det = pred_det_instances[new_tracks_mask]

        # create new id for new tracks
        num_new_tracks = len(new_det)
        new_ids = torch.arange(
            self.num_tracks,
            self.num_tracks + num_new_tracks,
            dtype=torch.long)
        new_det.ids = new_ids
        self.num_tracks += num_new_tracks

        # merge the two
        if existing_tracks_mask.sum() > 0:
            pred_track_instances = InstanceData.cat([existing_tracks, new_det])
            track_static_query_ids = torch.cat((
                existing_tracks_static_query_ids,
                new_det.query_ids,
            ),
                                               dim=0)
        else:
            pred_track_instances = new_det
            track_static_query_ids = new_det.query_ids

        # perform non-maxima suppression on the total,
        # force keeping the existing tracks
        if self.idol_nms:
            keep_inds = mask_nms(
                pred_track_instances.masks,
                pred_track_instances.scores,
                # labels=labels,
                nms_thr=0.4,
                white_list=None if existing_tracks_mask.sum() == 0 else list(
                    range(len(existing_tracks.ids))))
            pred_track_instances = pred_track_instances[keep_inds]
        else:
            scores, labels, masks, keep_inds = mask_matrix_nms(
                pred_track_instances.masks,
                pred_track_instances.labels,
                pred_track_instances.scores.to(
                    pred_track_instances.masks.device),
                nms_pre=self.nms_cfg.nms_pre,
                max_num=self.nms_cfg.max_per_img,
                kernel=self.nms_cfg.kernel,
                sigma=self.nms_cfg.sigma,
                filter_thr=0.35)
            keep_inds = keep_inds.to('cpu')
            pred_track_instances = pred_track_instances[keep_inds]
            pred_track_instances.scores = scores
            pred_track_instances.labels = labels
            pred_track_instances.masks = masks
        track_static_query_ids = track_static_query_ids[keep_inds]

        self.update(
            ids=pred_track_instances.ids,
            feats=pred_track_instances.feats,
            bboxes=pred_track_instances.bboxes,
            labels=pred_track_instances.labels,
            masks=pred_track_instances.masks,
            frame_ids=frame_id,
            track_static_query_ids=track_static_query_ids,
        )
        pred_track_instances.instances_id = pred_track_instances.ids
        return pred_track_instances


# adopted from the IDOL paper
# https://github.com/wjf5203/VNext/search?p=5&q=nms


def mask_iou(mask1, mask2):
    mask1 = mask1.char()
    mask2 = mask2.char()

    intersection = (mask1 * mask2).sum(-1).sum(-1)
    union = (mask1 + mask2 - mask1 * mask2).sum(-1).sum(-1)

    return (intersection + 1e-6) / (union + 1e-6)


def mask_nms(seg_masks, scores, labels=None, nms_thr=0.5, white_list=None):
    n_samples = len(scores)
    assert n_samples == len(seg_masks)
    if labels is not None:
        assert n_samples == len(labels)

    if white_list is None:
        white_list = []
    else:
        assert len(white_list) <= n_samples

    if n_samples == 0:
        return []
    keep = [True for i in range(n_samples)]

    if not (seg_masks.dtype == torch.bool):
        seg_masks = seg_masks.sigmoid() > 0.5

    for i in range(n_samples - 1):
        if not keep[i]:
            continue
        mask_i = seg_masks[i]
        # label_i = cate_labels[i]
        for j in range(i + 1, n_samples, 1):
            if j in white_list:
                continue
            if not keep[j]:
                continue
            if labels is not None and labels[i] != labels[j]:
                continue
            mask_j = seg_masks[j]

            iou = mask_iou(mask_i, mask_j)
            if iou > nms_thr:
                keep[j] = False
    return keep
