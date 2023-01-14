# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict

import torch
import torch.nn.functional as F
from mmdet.structures.mask.utils import mask2bbox
from mmengine.structures import InstanceData

from mmtrack.models.mot import BaseMultiObjectTracker
from mmtrack.models.track_heads.rovis_head import tensor_deselect
from mmtrack.registry import MODELS
from mmtrack.structures.track_data_sample import TrackDataSample
from mmtrack.utils.typing import OptConfigType, SampleList


def get_ref_data_samples(data_samples):
    ref_data_samples = []
    for data_sample in data_samples:
        ref_data_samples.append(
            TrackDataSample(
                gt_instances=data_sample.ref_gt_instances,
                metainfo=data_sample.metainfo))
    return ref_data_samples


def full_mask_matching(assigner, sampler, cls_score, mask_pred, gt_instances,
                       img_metas):
    """Compute classification and mask targets for one image.

    Args:
        cls_score (Tensor): Mask score logits from a single decoder layer
            for one image. Shape (num_queries, cls_out_channels).
        mask_pred (Tensor): Mask logits for a single decoder layer for one
            image. Shape (num_queries, h, w).
        gt_labels (Tensor): Ground truth class indices for one image with
            shape (n, ). n is the sum of number of stuff type and number
            of instance in a image.
        gt_masks (Tensor): Ground truth mask for each image, each with
            shape (n, h, w).
        img_metas (dict): Image informtation.

    Returns:
        tuple[Tensor]: a tuple containing the following for one image.
            - labels (Tensor): Labels of each image.
                shape (num_queries, ).
            - label_weights (Tensor): Label weights of each image.
                shape (num_queries, ).
            - mask_targets (Tensor): Mask targets of each image.
                shape (num_queries, h, w).
            - mask_weights (Tensor): Mask weights of each image.
                shape (num_queries, ).
            - pos_inds (Tensor): Sampled positive indices for each image.
            - neg_inds (Tensor): Sampled negative indices for each image.
    """
    # check
    target_shape = mask_pred.shape[-2:]
    gt_masks = gt_instances.masks
    if gt_masks.shape[0] > 0:
        gt_masks_downsampled = F.interpolate(
            gt_masks.unsqueeze(1).float(), target_shape,
            mode='nearest').squeeze(1).long()
    else:
        gt_masks_downsampled = gt_masks

    gt_instances_downsampled = InstanceData(
        labels=gt_instances.labels, masks=gt_masks_downsampled)
    pred_instances = InstanceData(scores=cls_score, masks=mask_pred)

    # assign and sample
    assign_result = assigner.assign(
        pred_instances=pred_instances,
        gt_instances=gt_instances_downsampled,
        img_meta=img_metas)
    sampling_result = sampler.sample(
        assign_result=assign_result,
        pred_instances=pred_instances,
        gt_instances=gt_instances_downsampled)
    pos_inds = sampling_result.pos_inds

    return pos_inds.cpu(), sampling_result.pos_assigned_gt_inds.cpu()


@MODELS.register_module()
class ROVIS(BaseMultiObjectTracker):
    """Video Instance Segmentation.

    Args:
        detector (dict): Configuration of detector. Defaults to None.
        track_head (dict): Configuration of track head. Defaults to None.
        tracker (dict): Configuration of tracker. Defaults to None.
        init_cfg (dict): Configuration of initialization. Defaults to None.
    """

    def __init__(self,
                 detector=None,
                 tracker=None,
                 data_preprocessor: OptConfigType = None,
                 query_substitute=False,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 **kwargs):
        super().__init__(data_preprocessor, init_cfg)

        if detector is not None:
            self.detector = MODELS.build(detector)
        else:
            print('No detector is specified.')
            exit()

        assert train_cfg is not None
        assert 'track_query_false_positive_prob' in train_cfg
        assert 'track_query_false_negative_prob' in train_cfg
        self._track_query_false_positive_prob = train_cfg[
            'track_query_false_positive_prob']
        self._track_query_false_negative_prob = train_cfg[
            'track_query_false_negative_prob']

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if tracker is not None:
            self.tracker = MODELS.build(tracker)

        self.query_substitute = query_substitute

    def add_track_queries_to_targets(self,
                                     prev_indices,
                                     prev_masks,
                                     data_samples,
                                     add_false_pos=True,
                                     length_padding=True):
        """select track queries for next frame's targets.

        targets: dict, {}
        prev_indices: [(Tensor, Tensor),]  contains the matching results
            prev_out_ind, prev_target_ind
        return:
            selected previous out indices, selected current target indices
            negative means no match
        """
        res = []

        for batch_idx, (prev_ind, data_sample) in enumerate(
                zip(prev_indices, data_samples)):
            prev_instance_ids_per_img = data_sample.gt_instances.instances_id
            instance_ids_per_img = data_sample.ref_gt_instances.instances_id

            prev_out_ind, prev_target_ind = prev_ind
            prev_instance_ids_per_img = prev_instance_ids_per_img.to(
                prev_out_ind.device)
            instance_ids_per_img = instance_ids_per_img.to(prev_out_ind.device)

            # random subset
            if self._track_query_false_negative_prob:
                random_subset_mask = torch.rand(
                    (len(prev_out_ind),
                     )) < self._track_query_false_negative_prob

                prev_out_ind = prev_out_ind[random_subset_mask]
                prev_target_ind = prev_target_ind[random_subset_mask]

            # detected prev frame tracks
            prev_track_ids = prev_instance_ids_per_img[prev_target_ind]

            # match track ids between frames
            target_ind_match_matrix = prev_track_ids.unsqueeze(dim=1).eq(
                instance_ids_per_img)  # (num_prev, num_cur)
            target_ind_matching = target_ind_match_matrix.any(
                dim=1)  # in the previous frame proposal being matched
            target_ind_matched_idx = target_ind_match_matrix.nonzero(
            )[:, 1]  # which ids in the current frame are being matched

            # current frame track ids detected in the prev frame
            # track_ids = target['track_ids'][target_ind_matched_idx]

            # index of prev frame detection in current frame box list
            # target['track_query_match_ids'] = target_ind_matched_idx

            # random false positives
            if add_false_pos:
                not_prev_out_ind = torch.arange(prev_masks.shape[1])
                not_prev_out_ind = tensor_deselect(not_prev_out_ind,
                                                   prev_ind[0]).tolist()

                random_false_out_ind = []

                prev_target_ind_for_fps = torch.arange(
                    len(prev_out_ind))[torch.rand((len(prev_out_ind), )) <
                                       self._track_query_false_positive_prob]

                for j in prev_target_ind_for_fps:
                    prev_masks_unmatched = prev_masks[batch_idx,
                                                      not_prev_out_ind]

                    if target_ind_matching[j]:
                        prev_bbox_matched = mask2bbox(
                            prev_masks[batch_idx,
                                       prev_out_ind[j]].unsqueeze(0) > 0)
                        box_weights = \
                            prev_bbox_matched[:, :2] - \
                            mask2bbox(prev_masks_unmatched > 0)[:, :2]
                        box_weights = box_weights[:, 0]**2 + box_weights[:,
                                                                         1]**2
                        box_weights = torch.sqrt(box_weights)
                        try:
                            random_false_out_idx = not_prev_out_ind.pop(
                                torch.multinomial(box_weights.cpu(), 1).item())
                        except Exception:
                            # in rare cases the weights might be all zeros
                            random_false_out_idx = not_prev_out_ind.pop(
                                torch.randperm(len(not_prev_out_ind))[0])
                            print(box_weights)
                    else:
                        random_false_out_idx = not_prev_out_ind.pop(
                            torch.randperm(len(not_prev_out_ind))[0])

                    random_false_out_ind.append(random_false_out_idx)

                prev_out_ind = torch.tensor(prev_out_ind.tolist() +
                                            random_false_out_ind).long()
                # makes sure that all the true negatives are inside
                target_ind_matching = torch.cat([
                    target_ind_matching,
                    torch.tensor([
                        False,
                    ] * len(random_false_out_ind)).bool()
                ])

            assert len(prev_out_ind) == len(
                target_ind_matching), "prev_out_ind and target_ind_matching \
                must have the same length. Note that target_ind_matching is a \
                boolean tensor to mask which \
                prev_out_ind actually matches with the current frame's target"

            target_ind = prev_out_ind.new_tensor([
                -1,
            ] * len(prev_out_ind))
            target_ind[target_ind_matching] = target_ind_matched_idx
            res.append((prev_out_ind, target_ind))
            continue

        if not length_padding:
            return res

        # pad all the matching with garbage backgrounds
        longest_len = max([len(i[0]) for i in res])
        for batch_idx, prev_ind in enumerate(prev_indices, ):
            prev_out_ind, target_ind = res[batch_idx]
            num_selectable = prev_masks.shape[1]
            if len(prev_out_ind) < longest_len:
                # shorter than expected, generate some padding
                selection = torch.randint(
                    low=0,
                    high=num_selectable,
                    size=(longest_len - len(prev_out_ind), ),
                )
                cannot_select = set(prev_out_ind.tolist()) | set(
                    prev_ind[0].tolist())
                while set(selection.tolist()) & cannot_select:
                    selection = torch.randint(
                        low=0,
                        high=num_selectable,
                        size=(longest_len - len(prev_out_ind), ))
                # selected padding candidate, now concat
                prev_out_ind = torch.cat([prev_out_ind, selection])
                target_ind = torch.cat([
                    target_ind,
                    torch.tensor([-1] * len(selection)),
                ], )
                assert len(prev_out_ind) == len(target_ind)
                res[batch_idx] = (prev_out_ind, target_ind)
        return res

    def loss(self, inputs: Dict[str, torch.Tensor], data_samples: SampleList,
             **kwargs):
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Dict[str, Tensor]): of shape (N, T, C, H, W) encoding
                input images. Typically these should be mean centered and std
                scaled. The N denotes batch size.The T denotes the number of
                key/reference frames.
                - img (Tensor) : The key images.
                - ref_img (Tensor): The reference images.
            data_samples (list[:obj:`TrackDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance`.

        Returns:
            dict: A dictionary of loss components.
        """
        # modify the inputs shape to fit mmdet
        img = inputs['img']
        assert img.dim() == 5, 'The img must be 5D Tensor (N, T, C, H, W).'
        assert img.size(1) == 1, \
            'MaskTrackRCNN can only have 1 key frame and 1 reference frame.'
        img = img[:, 0]

        ref_img = inputs['ref_img']
        assert ref_img.dim() == 5, 'The img must be 5D Tensor (N, T, C, H, W).'
        assert ref_img.size(1) == 1, \
            'MaskTrackRCNN can only have 1 key frame and 1 reference frame.'
        ref_img = ref_img[:, 0]

        # batch processing for img and ref img
        x = self.detector.extract_feat(img)
        ref_x = self.detector.extract_feat(ref_img)

        losses = dict()

        if hasattr(self.detector,
                   'mask_head'):  # for SOLO and MaskRCNN, this won't be used
            mask_loss = self.detector.mask_head.forward_train(
                x, data_samples, **kwargs)
        elif hasattr(self.detector,
                     'panoptic_head'):  # for MaskFormer class model
            mask_loss = self.detector.panoptic_head.loss(x, data_samples)
            # extract inference results
            all_mask_preds = mask_loss.pop('all_mask_pred')
            all_cls_scores = mask_loss.pop('all_cls_scores')
            last_query_feats = mask_loss.pop('last_query_feats')

            # detach
            # last_query_feats = last_query_feats

            processed_gt = mask_loss.pop('gt')

            with torch.no_grad():
                # match last mask prediction to ground truth
                prev_indices = [
                    full_mask_matching(
                        self.detector.panoptic_head.assigner,
                        self.detector.panoptic_head.sampler,
                        all_cls_scores[-1][i],
                        all_mask_preds[-1][i],
                        processed_gt[i],
                        data_samples[i].metainfo,
                    ) for i in range(len(data_samples))
                ]
                # prepare the queries and ground truth for the reference frame
                cross_frame_matching = self.add_track_queries_to_targets(
                    prev_indices,
                    all_mask_preds[-1],
                    data_samples,
                    length_padding=False if self.query_substitute else True)
                additional_queries = []
                additional_query_target_index = []
                prev_query_indices = []
                for batch_idx, (
                        prev_query_idx,
                        curr_target_idx) in enumerate(cross_frame_matching):
                    additional_queries.append(last_query_feats[prev_query_idx,
                                                               batch_idx,
                                                               ...].detach())
                    additional_query_target_index.append(curr_target_idx)
                    prev_query_indices.append(
                        prev_query_idx)  # only useful in substitution mode

            # inference on reference images
            ref_mask_loss = self.detector.panoptic_head.loss(
                ref_x,
                get_ref_data_samples(data_samples),
                additional_queries=additional_queries,
                additional_query_target_indices=additional_query_target_index,
                remove_static_query=prev_query_indices
                if self.query_substitute else None,
                additional_query_embed=prev_query_indices
                if self.detector.panoptic_head.use_query_embed else None,
            )
            # clean inference results
            _ = ref_mask_loss.pop('all_mask_pred')
            _ = ref_mask_loss.pop('all_cls_scores')
            _ = ref_mask_loss.pop('last_query_feats')

            # add prefix for ref_mask_loss
            # divide by 2
            ref_mask_loss = {
                'ref_' + k: v / 2.0
                for k, v in ref_mask_loss.items()
            }
            assert not set(ref_mask_loss.keys()) & set(losses.keys())
            losses.update(ref_mask_loss)

        # avoid loss override
        assert not set(mask_loss.keys()) & set(losses.keys())
        losses.update(mask_loss)

        return losses

    def predict(self,
                inputs: Dict[str, torch.Tensor],
                data_samples: SampleList,
                rescale=False,
                track_maintain_conf=0.25,
                track_est_conf=0.3,
                **kwargs):
        """Test without augmentations.
        Args:
            img (Tensor): of shape (1, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            rescale (bool, optional): If False, then returned bboxes and masks
                will fit the scale of img, otherwise, returned bboxes and masks
                will fit the scale of original image shape. Defaults to False.

        Returns:
            dict[str : list(ndarray)]: The tracking results.
        """
        img = inputs['img']
        assert img.dim() == 5, 'The img must be 5D Tensor (N, T, C, H, W).'
        assert img.size(1) == 1, \
            'rovis can only have 1 key frame.'
        img = img[:, 0]

        assert len(data_samples) == 1, \
            'roivs only support 1 batch size per gpu for now.'

        metainfo = data_samples[0].metainfo
        frame_id = metainfo.get('frame_id', -1)

        if frame_id == 0:
            self.tracker.reset()
            input_queries = None
            track_static_query_ids = None
        elif self.tracker.empty:
            input_queries = None
            track_static_query_ids = None
        else:
            input_queries = self.tracker.get('feats')
            track_static_query_ids = self.tracker.get('track_static_query_ids')
        num_queries = self.detector.panoptic_head.num_queries

        # extract features
        x = self.detector.extract_feat(img)

        # forward Mask2Former
        if hasattr(self.detector, 'panoptic_head'):
            # copy from maskformer simple_test
            # img_metas[0]['batch_input_shape'] = img_metas[0]['img_shape']
            (mask_cls_results, mask_pred_results,
             queries) = self.detector.panoptic_head.predict(
                 x,
                 data_samples,
                 remove_static_query=[track_static_query_ids] if
                 (self.query_substitute
                  and track_static_query_ids is not None) else None,
                 additional_query_embed=[track_static_query_ids] if
                 (self.detector.panoptic_head.use_query_embed
                  and track_static_query_ids is not None) else None,
                 additional_queries=[input_queries]
                 if input_queries is not None else None,
             )
            # some work around of handling dynamic number of queries.
            self.detector.panoptic_fusion_head.test_cfg[
                'max_per_image'] = mask_cls_results.shape[1]
            results = self.detector.panoptic_fusion_head.predict(
                mask_cls_results,
                mask_pred_results,
                data_samples,
                rescale=rescale)

            # queries returned has shape (num_query, batch_size, channel)
            queries = queries[:, 0, ...]

            assert len(results) == 1
            assert 'ins_results' in results[0]
            labels_per_image, bboxes, mask_pred_binary = results[0][
                'ins_results']
            # query id is in the fst idx, while confidence is in the last idx
            scores = bboxes[:, -1].cpu()
            query_ids = bboxes[:, 0].cpu().long()
            bboxes = bboxes[:, 1:]  # remove query_id

            # apply the permutation to queries also
            queries = queries[query_ids]

            if self.query_substitute and input_queries is not None:
                # find the original query id
                original_query_ids = torch.arange(
                    num_queries + len(input_queries), device='cpu')
                # evict the static query, afterwards idx correspond to queries
                original_query_ids = tensor_deselect(original_query_ids,
                                                     track_static_query_ids)
                query_ids = original_query_ids[query_ids]

            # pack into instance data
            pred_track_instances = InstanceData()
            pred_track_instances.feats = queries
            pred_track_instances.query_ids = query_ids
            pred_track_instances.bboxes = bboxes
            pred_track_instances.labels = labels_per_image
            pred_track_instances.masks = mask_pred_binary
            pred_track_instances.scores = scores

            # do filtering by confidence
            keep = (torch.logical_or(
                torch.logical_and(scores > track_maintain_conf,
                                  query_ids >= num_queries),
                torch.logical_and(
                    scores > track_est_conf,
                    query_ids < num_queries))).nonzero(as_tuple=True)[0]
            pred_track_instances = pred_track_instances[keep]

            # sort by confidence
            _, sort_idx = torch.sort(
                pred_track_instances.scores, descending=True)
            pred_track_instances = pred_track_instances[sort_idx]
        else:
            raise NotImplementedError
        track_data_sample = data_samples[0]
        track_data_sample.pred_det_instances = pred_track_instances
        pred_track_instances = self.tracker.track(
            img=img,
            img_metas=[metainfo],
            model=self,
            data_sample=track_data_sample,
            frame_id=frame_id,
            rescale=False,  # this has already been rescaled
            **kwargs)

        track_data_sample.pred_track_instances = pred_track_instances
        return [track_data_sample]
