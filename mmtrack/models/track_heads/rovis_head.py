# Copyright (c) OpenMMLab. All rights reserved.
"""MaskFormerFusionHead Some modification of the original code from:

https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/dense_heads/maskformer_head.py
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from mmcv.ops import point_sample
from mmdet.models.dense_heads.mask2former_head import Mask2FormerHead
from mmdet.models.seg_heads.panoptic_fusion_heads import MaskFormerFusionHead
from mmdet.models.utils import get_uncertain_point_coords_with_randomness
from mmdet.structures import SampleList
from mmdet.structures.mask.utils import mask2bbox
from mmdet.utils import reduce_mean
from mmengine.structures import InstanceData

from mmtrack.registry import MODELS


def tensor_deselect(tensor: torch.Tensor, indices):
    mask = torch.ones(tensor.shape[0]).bool()
    mask[indices] = False
    if len(tensor[mask]) == 0:
        # create a dummy tensor
        return tensor.new_zeros((0, ) + tensor.shape[1:])
    return tensor[mask]


def deselect_mask(len_mask, indices):
    mask = torch.ones(len_mask, dtype=torch.bool)
    mask[indices] = False
    return mask


@MODELS.register_module()
class AdditionalQueriesFormerHead(Mask2FormerHead):

    def __init__(self,
                 use_query_embed=False,
                 disable_embed=False,
                 *args,
                 **kwargs):
        super(AdditionalQueriesFormerHead, self).__init__(*args, **kwargs)
        self.use_query_embed = use_query_embed
        self.disable_embed = disable_embed
        if disable_embed:
            with torch.no_grad():
                self.query_embed.weight[...] = 0
            for p in self.query_embed.parameters():
                p.requires_grad_(False)

    def _loss_by_feat_single(
            self, cls_scores: torch.Tensor, mask_preds: torch.Tensor,
            batch_gt_instances: List[InstanceData],
            batch_img_metas: List[dict]) -> Tuple[torch.Tensor]:
        """Loss function for outputs from a single decoder layer. Modified from
        `mmdet/models/dense_heads/mask2former_head.py`. The only modification
        is allowing cls_scores and mask_preds to be a list of tensors.

        Args:
            cls_scores (Tensor): Mask score logits from a single decoder layer
                for all images. Shape (batch_size, num_queries,
                cls_out_channels). Note `cls_out_channels` should includes
                background.
            mask_preds (Tensor): Mask logits for a pixel decoder for all
                images. Shape (batch_size, num_queries, h, w).
            batch_gt_instances (list[obj:`InstanceData`]): each contains
                ``labels`` and ``masks``.
            batch_img_metas (list[dict]): List of image meta information.

        Returns:
            tuple[Tensor]: Loss components for outputs from a single \
                decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        mask_preds_list = [mask_preds[i] for i in range(num_imgs)]
        (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
         avg_factor) = self.get_targets(cls_scores_list, mask_preds_list,
                                        batch_gt_instances, batch_img_metas)

        # for substitution mode: truncate the length
        labels_list = [
            labels_list[i][:len(cls_scores_list[i])] for i in range(num_imgs)
        ]
        label_weights_list = [
            label_weights_list[i][:len(cls_scores_list[i])]
            for i in range(num_imgs)
        ]
        mask_weights_list = [
            mask_weights_list[i][:len(cls_scores_list[i])]
            for i in range(num_imgs)
        ]

        # shape (batch_size, num_queries)
        labels = torch.cat(labels_list, dim=0)
        # shape (batch_size, num_queries)
        label_weights = torch.cat(label_weights_list, dim=0)
        # shape (num_total_gts, h, w)
        mask_targets = torch.cat(mask_targets_list, dim=0)
        # shape (batch_size, num_queries)
        mask_weights = torch.cat(mask_weights_list, dim=0)
        mask_preds = torch.cat(mask_preds_list, dim=0)
        # classfication loss
        # shape (batch_size * num_queries, )
        cls_scores = torch.cat(cls_scores_list)
        # labels = labels.flatten(0, 1)
        # label_weights = label_weights.flatten(0, 1)

        class_weight = cls_scores.new_tensor(self.class_weight)
        loss_cls = self.loss_cls(
            cls_scores,
            labels,
            label_weights,
            avg_factor=class_weight[labels].sum())

        num_total_masks = reduce_mean(cls_scores.new_tensor([avg_factor]))
        num_total_masks = max(num_total_masks, 1)

        # extract positive ones
        # shape (batch_size, num_queries, h, w) -> (num_total_gts, h, w)
        mask_preds = mask_preds[mask_weights > 0]

        if mask_targets.shape[0] == 0:
            # zero match
            loss_dice = mask_preds.sum()
            loss_mask = mask_preds.sum()
            return loss_cls, loss_mask, loss_dice

        with torch.no_grad():
            points_coords = get_uncertain_point_coords_with_randomness(
                mask_preds.unsqueeze(1), None, self.num_points,
                self.oversample_ratio, self.importance_sample_ratio)
            # shape (num_total_gts, h, w) -> (num_total_gts, num_points)
            mask_point_targets = point_sample(
                mask_targets.unsqueeze(1).float(), points_coords).squeeze(1)
        # shape (num_queries, h, w) -> (num_queries, num_points)
        mask_point_preds = point_sample(
            mask_preds.unsqueeze(1), points_coords).squeeze(1)

        # dice loss
        loss_dice = self.loss_dice(
            mask_point_preds, mask_point_targets, avg_factor=num_total_masks)

        # mask loss
        # shape (num_queries, num_points) -> (num_queries * num_points, )
        mask_point_preds = mask_point_preds.reshape(-1)
        # shape (num_total_gts, num_points) -> (num_total_gts * num_points, )
        mask_point_targets = mask_point_targets.reshape(-1)
        loss_mask = self.loss_mask(
            mask_point_preds,
            mask_point_targets,
            avg_factor=num_total_masks * self.num_points)

        return loss_cls, loss_mask, loss_dice

    def matched_loss_single(
        self,
        cls_scores_list,
        mask_preds_list,
        gt_labels_list,
        gt_masks_list,
    ):
        """Assumed that predictions and targets have already been matched. Loss
        function for outputs from a single decoder layer.

        Args:
            cls_scores (Tensor): Mask score logits from a single decoder layer
                for all images. Shape (batch_size, num_queries,
                cls_out_channels). Note `cls_out_channels` should includes
                background.
            mask_preds (Tensor): Mask logits for a pixel decoder for all
                images. Shape (batch_size, num_queries, h, w).
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image, each with shape (num_gts, ).
            gt_masks_list (list[Tensor]): Ground truth mask for each image,
                each with shape (num_gts, h, w).
            img_metas (list[dict]): List of image meta information.

        Returns:
            tuple[Tensor]: Loss components for outputs from a single \
                decoder layer.
        """
        cls_scores = torch.cat(cls_scores_list)
        labels = torch.cat(gt_labels_list)
        label_weights = torch.ones_like(labels)
        mask_targets = torch.cat(gt_masks_list)
        mask_preds = torch.cat(mask_preds_list)

        assert cls_scores.shape[0] == labels.shape[0], str(
            cls_scores.shape[0]) + ' ' + str(labels.shape[0])
        assert mask_targets.shape[0] == mask_preds.shape[0], str(
            mask_targets.shape[0]) + ' ' + str(mask_preds.shape[0])

        class_weight = cls_scores.new_tensor(self.class_weight)
        loss_cls = self.loss_cls(
            cls_scores,
            labels,
            label_weights,
            avg_factor=class_weight[labels].sum())

        num_total_pos = mask_targets.shape[0]
        num_total_masks = reduce_mean(cls_scores.new_tensor([num_total_pos]))
        num_total_masks = max(num_total_masks, 1)

        if mask_targets.shape[0] == 0:
            # zero match
            loss_dice = mask_preds.sum()
            loss_mask = mask_preds.sum()
            return loss_cls, loss_mask, loss_dice

        with torch.no_grad():
            points_coords = get_uncertain_point_coords_with_randomness(
                mask_preds.unsqueeze(1), None, self.num_points,
                self.oversample_ratio, self.importance_sample_ratio)
            # shape (num_total_gts, h, w) -> (num_total_gts, num_points)
            mask_point_targets = point_sample(
                mask_targets.unsqueeze(1).float(), points_coords).squeeze(1)
        # shape (num_queries, h, w) -> (num_queries, num_points)
        mask_point_preds = point_sample(
            mask_preds.unsqueeze(1), points_coords).squeeze(1)

        # dice loss
        loss_dice = self.loss_dice(
            mask_point_preds, mask_point_targets, avg_factor=num_total_masks)

        # mask loss
        # shape (num_queries, num_points) -> (num_queries * num_points, )
        mask_point_preds = mask_point_preds.reshape(-1)
        # shape (num_total_gts, num_points) -> (num_total_gts * num_points, )
        mask_point_targets = mask_point_targets.reshape(-1)
        loss_mask = self.loss_mask(
            mask_point_preds,
            mask_point_targets,
            avg_factor=num_total_masks * self.num_points)

        return loss_cls, loss_mask, loss_dice

    # override the original forward function
    def forward(
        self,
        x,
        batch_data_samples: SampleList,
        additional_queries=None,
        additional_query_embed=None,
        additional_query_only=False,
        return_attn_map=False,
        remove_static_query=None,
    ):
        """Forward function. What's different from the original Mask2FormerHead
        is that the additional_queries are added to the input.

        Args:
            feats (list[Tensor]): Multi scale Features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.
            additional_queries (list[Tensor]): Additional queries for the
                forward function. Inner tensor has shape (num_gts, emb_dim).

        Returns:
            tuple: A tuple contains two elements.

            - cls_pred_list (list[Tensor)]: Classification logits \
                for each decoder layer. Each is a 3D-tensor with shape \
                (batch_size, num_queries, cls_out_channels). \
                Note `cls_out_channels` should includes background.
            - mask_pred_list (list[Tensor]): Mask logits for each \
                decoder layer. Each with shape (batch_size, num_queries, \
                 h, w).
        """
        batch_img_metas = [
            data_sample.metainfo for data_sample in batch_data_samples
        ]
        batch_size = len(batch_img_metas)

        if self.use_query_embed and additional_queries is not None:
            assert additional_query_embed is not None
        if additional_query_embed is not None and additional_query_embed[
                0].ndim == 1:
            # given additional query embed as indices, take the raw embeddings
            additional_query_embed = [
                self.query_embed.weight[x] for x in additional_query_embed
            ]

        mask_features, multi_scale_memorys = self.pixel_decoder(x)
        # multi_scale_memorys (from low resolution to high resolution)
        decoder_inputs = []
        decoder_positional_encodings = []
        for i in range(self.num_transformer_feat_level):
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
            # shape (batch_size, c, h, w) -> (h*w, batch_size, c)
            decoder_input = decoder_input.flatten(2).permute(2, 0, 1)
            level_embed = self.level_embed.weight[i].view(1, 1, -1)
            decoder_input = decoder_input + level_embed
            # shape (batch_size, c, h, w) -> (h*w, batch_size, c)
            mask = decoder_input.new_zeros(
                (batch_size, ) + multi_scale_memorys[i].shape[-2:],
                dtype=torch.bool)
            decoder_positional_encoding = self.decoder_positional_encoding(
                mask)
            decoder_positional_encoding = decoder_positional_encoding.flatten(
                2).permute(2, 0, 1)
            decoder_inputs.append(decoder_input)
            decoder_positional_encodings.append(decoder_positional_encoding)
        # shape (num_queries, c) -> (num_queries, batch_size, c)
        if remove_static_query is not None:
            query_feat = self.query_feat.weight
            query_embed = self.query_embed.weight
        elif additional_query_only:  # set of static queries is empty
            query_feat = self.query_feat.weight.new_zeros(
                (0, batch_size, self.query_feat.weight.shape[-1]))
            query_embed = self.query_embed.weight.new_zeros(
                (0, batch_size, self.query_embed.weight.shape[-1]))
        else:  # the original case is query_feat is copied batch_size times
            query_feat = self.query_feat.weight.unsqueeze(1).repeat(
                (1, batch_size, 1))
            query_embed = self.query_embed.weight.unsqueeze(1).repeat(
                (1, batch_size, 1))

        # handle additional queries
        if additional_queries is None:
            # create a dummy additional_queries
            additional_queries = [
                torch.zeros((0, query_feat.shape[-1]),
                            device=query_feat.device)
                for _ in range(batch_size)
            ]

        if remove_static_query is not None:
            query_embed_batch = []
            # first concat additional queries with valid static queries
            for i in range(batch_size):
                additional_queries[i] = torch.cat([
                    tensor_deselect(query_feat, remove_static_query[i]),
                    additional_queries[i],
                ],
                                                  dim=0)
                deselected_query_embed = tensor_deselect(
                    query_embed, remove_static_query[i])
                if additional_query_embed is not None:
                    additional_query_embed_i = additional_query_embed[i]
                else:
                    additional_query_embed_i = \
                        self.query_embed.weight.new_zeros(
                            (additional_queries[i].shape[0] -
                                deselected_query_embed.shape[0],
                                self.query_embed.weight.shape[-1]))
                query_embed_batch.append(
                    torch.cat([
                        deselected_query_embed,
                        additional_query_embed_i,
                    ],
                              dim=0))
            query_embed = torch.stack(query_embed_batch, dim=1)
        longest_query_len = max([
            additional_queries[i].shape[0]
            for i in range(len(additional_queries))
        ])

        # (batch, query_num, emb_dim) -> (query_num, batch, emb_dim)
        additional_queries = torch.stack(
            additional_queries, dim=0).permute(1, 0, 2)

        # concat additional queries to the query_feat
        if remove_static_query is None:  # this is the original case
            query_feat = torch.cat((query_feat, additional_queries), dim=0)

            if additional_query_embed is None:
                # create query_embed_dumb for additional queries
                # that will be added to the query embed
                query_embed_dummy = torch.zeros(
                    (longest_query_len, batch_size, query_embed.shape[-1]),
                    dtype=query_embed.dtype,
                    device=query_embed.device)
            else:
                # use the additional_query_embed
                # (query_num, emb_dim) -> (query_num, batch, emb_dim)
                query_embed_dummy = torch.stack(
                    additional_query_embed, dim=0).permute(1, 0, 2)
            # concat the additional queries emb to the query_embed
            query_embed = torch.cat((query_embed, query_embed_dummy), dim=0)
        else:
            query_feat = additional_queries
        # store the cross attention maps
        crossattn_maps = []
        multi_scale_resolutions = [
            multi_scale_memorys[i].shape[-2:]
            for i in range(self.num_transformer_feat_level)
        ]

        cls_pred_list = []
        mask_pred_list = []
        cls_pred, mask_pred, attn_mask = self._forward_head(
            query_feat, mask_features, multi_scale_memorys[0].shape[-2:])
        cls_pred_list.append(cls_pred)
        mask_pred_list.append(mask_pred)

        for i in range(self.num_transformer_decoder_layers):
            level_idx = i % self.num_transformer_feat_level
            # if a mask is all True(all background), then set it all False.
            attn_mask[torch.where(
                attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            # cross_attn + self_attn
            layer = self.transformer_decoder.layers[i]
            attn_masks = [attn_mask, None]
            query_feat = layer(
                query=query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=query_embed if not self.disable_embed else None,
                key_pos=decoder_positional_encodings[level_idx],
                attn_masks=attn_masks,
                query_key_padding_mask=None,
                # here we do not apply masking on padded region
                key_padding_mask=None,
                return_attn_map=return_attn_map)
            if return_attn_map:
                # pop the cross attention map
                assert type(query_feat) is tuple
                crossattn_maps.append(query_feat[1][0])
                query_feat = query_feat[0]
            cls_pred, mask_pred, attn_mask = self._forward_head(
                query_feat, mask_features, multi_scale_memorys[
                    (i + 1) % self.num_transformer_feat_level].shape[-2:])

            cls_pred_list.append(cls_pred)
            mask_pred_list.append(mask_pred)

        if return_attn_map:
            return (cls_pred_list, mask_pred_list, query_feat, crossattn_maps,
                    multi_scale_resolutions)
        return cls_pred_list, mask_pred_list, query_feat

    def loss(self,
             x,
             batch_data_samples: SampleList,
             additional_queries=None,
             additional_query_target_indices=None,
             additional_query_only=False,
             remove_static_query: Optional[List] = None,
             **kwargs):
        """Forward function for training mode.

        Args:
            feats (list[Tensor]): Multi-level features from the upstream
                network, each is a 4D-tensor.
            img_metas (list[Dict]): List of image information.
            gt_bboxes (list[Tensor]): Each element is ground truth bboxes of
                the image, shape (num_gts, 4). Not used here.
            gt_labels (list[Tensor]): Each element is ground truth labels of
                each box, shape (num_gts,).
            gt_masks (list[BitmapMasks]): Each element is masks of instances
                of a image, shape (num_gts, h, w).
            gt_semantic_seg (list[tensor]):Each element is the ground truth
                of semantic segmentation with the shape (N, H, W).
                [0, num_thing_class - 1] means things,
                [num_thing_class, num_class-1] means stuff,
                255 means VOID.
            gt_bboxes_ignore (list[Tensor]): Ground truth bboxes to be
                ignored. Defaults to None.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # not consider ignoring bboxes
        if additional_query_only:
            assert additional_queries is not None
            assert additional_query_target_indices is not None

        # for preprocessing, copied from mmdet
        batch_img_metas = []
        batch_gt_instances = []
        batch_gt_semantic_segs = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)
            if 'gt_sem_seg' in data_sample:
                batch_gt_semantic_segs.append(data_sample.gt_sem_seg)
                raise NotImplementedError
            else:
                batch_gt_semantic_segs.append(None)

        # forward
        all_cls_scores, all_mask_preds, last_query_feats = self(
            x,
            batch_data_samples,
            additional_queries=additional_queries,
            additional_query_only=additional_query_only,
            remove_static_query=remove_static_query,
            **kwargs)

        # preprocess ground truth
        batch_gt_instances = self.preprocess_gt(batch_gt_instances,
                                                batch_gt_semantic_segs)
        gt_labels = [gt.labels for gt in batch_gt_instances]
        gt_masks = [gt.masks for gt in batch_gt_instances]

        # exclude the targets of added queries from the loss auto-matching
        if additional_query_target_indices is not None:
            with torch.no_grad():
                additional_gt_labels = []
                additional_gt_masks = []
                for i in range(len(batch_img_metas)):
                    valid_indices = additional_query_target_indices[i][
                        additional_query_target_indices[i] >= 0]

                    additional_gt_label = torch.tensor(
                        [
                            self.num_classes,
                        ] * len(additional_query_target_indices[i]),
                        dtype=gt_labels[i].dtype)
                    # the entries with -1 target index are false positives
                    additional_gt_label[additional_query_target_indices[i] >=
                                        0] = gt_labels[i][valid_indices].to(
                                            additional_gt_label.device)
                    additional_gt_labels.append(
                        additional_gt_label.to(gt_labels[i].device))

                    additional_gt_masks.append(gt_masks[i][valid_indices])

                    if not additional_query_only:
                        # delete the targets of added queries from the gt
                        batch_gt_instances[i] = batch_gt_instances[i][
                            deselect_mask(
                                len(batch_gt_instances[i]), valid_indices)]
        # compute losses for the added queries
        additional_loss_dict = dict()
        if additional_query_target_indices is not None:
            for layer_id in range(len(all_cls_scores)):
                valid_masks = [
                    additional_query_target_indices[i] >= 0
                    for i in range(len(batch_gt_instances))
                ]  # for mask_preds, but also represented in binary values
                if additional_query_only:
                    cls_preds = [
                        all_cls_scores[layer_id][
                            i, :len(additional_queries[i])]
                        for i in range(len(batch_gt_instances))
                    ]
                elif remove_static_query is not None:
                    valid_masks = [
                        torch.cat([
                            valid_mask.new_tensor([
                                False,
                            ] * (self.num_queries -
                                 len(remove_static_query[batch_idx]))),
                            valid_mask
                        ]) for batch_idx, valid_mask in enumerate(valid_masks)
                    ]
                    cls_preds = [
                        all_cls_scores[layer_id][i, (
                            self.num_queries - len(remove_static_query[i])
                        ):((self.num_queries - len(remove_static_query[i])) +
                           len(additional_queries[i]))]
                        for i in range(len(batch_data_samples))
                    ]
                else:
                    valid_masks = [
                        torch.cat([
                            valid_mask.new_tensor([
                                False,
                            ] * self.num_queries), valid_mask
                        ]) for valid_mask in valid_masks
                    ]
                    cls_preds = [
                        all_cls_scores[layer_id][i, self.num_queries:(
                            self.num_queries + len(additional_queries[i]))]
                        for i in range(len(batch_gt_instances))
                    ]  # trim according to the length of additional_queries

                loss_cls, loss_mask, loss_dice = self.matched_loss_single(
                    cls_preds,
                    [
                        all_mask_preds[layer_id][i][valid_masks[i]]
                        for i in range(len(batch_img_metas))
                    ],  # trim according to mask available
                    additional_gt_labels,
                    additional_gt_masks,
                )
                # if not the last layer
                if layer_id < len(all_cls_scores) - 1:
                    additional_loss_dict[
                        f'd{layer_id}.loss_cls_add'] = loss_cls
                    additional_loss_dict[
                        f'd{layer_id}.loss_mask_add'] = loss_mask
                    additional_loss_dict[
                        f'd{layer_id}.loss_dice_add'] = loss_dice
                else:
                    additional_loss_dict['loss_cls_add'] = loss_cls
                    additional_loss_dict['loss_mask_add'] = loss_mask
                    additional_loss_dict['loss_dice_add'] = loss_dice

        if not additional_query_only:
            # loss
            if remove_static_query is None:
                all_cls_scores_static = [
                    i[:, :self.num_queries, ...] for i in all_cls_scores
                ]
                all_mask_preds_static = [
                    i[:, :self.num_queries, ...] for i in all_mask_preds
                ]
            else:
                all_cls_scores_static, all_mask_preds_static = SizeList(
                ), SizeList()
                for layer_id in range(len(all_cls_scores)):
                    all_cls_scores_static.append(SizeList())
                    all_mask_preds_static.append(SizeList())
                    for batch_idx, (i, j) in enumerate(
                            zip(all_cls_scores[layer_id],
                                all_mask_preds[layer_id])):
                        all_cls_scores_static[-1].append(
                            i[:self.num_queries -
                              len(remove_static_query[batch_idx])])
                        all_mask_preds_static[-1].append(
                            j[:self.num_queries -
                              len(remove_static_query[batch_idx])])
            losses = self.loss_by_feat(all_cls_scores_static,
                                       all_mask_preds_static,
                                       batch_gt_instances, batch_img_metas)
            # merge losses for added queries and assert no conflict
            assert not set(additional_loss_dict.keys()) & set(losses.keys())
            losses.update(additional_loss_dict)
        else:
            losses = {}
            losses.update(additional_loss_dict)

        # add inference results to losses dict
        losses['all_mask_pred'] = all_mask_preds
        losses['all_cls_scores'] = all_cls_scores
        losses['last_query_feats'] = last_query_feats
        if additional_query_target_indices is None:
            pass
            # pass the original gt_labels and gt_masks to the loss
            losses['gt'] = batch_gt_instances
        return losses

    def predict(self, x: Tuple[torch.Tensor], batch_data_samples: SampleList,
                **kwargs) -> Tuple[torch.Tensor]:
        """Copy and paste mask2former predict. The only difference is that is
        passes over the additional all_queries return value.

        Args:
            x (tuple[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            tuple[Tensor]: A tuple contains two tensors.

                - mask_cls_results (Tensor): Mask classification logits,\
                    shape (batch_size, num_queries, cls_out_channels).
                    Note `cls_out_channels` should includes background.
                - mask_pred_results (Tensor): Mask logits, shape \
                    (batch_size, num_queries, h, w).
        """
        batch_img_metas = [
            data_sample.metainfo for data_sample in batch_data_samples
        ]
        all_cls_scores, all_mask_preds, all_queries = self(
            x, batch_data_samples, **kwargs)
        mask_cls_results = all_cls_scores[-1]
        mask_pred_results = all_mask_preds[-1]

        # upsample masks
        img_shape = batch_img_metas[0]['batch_input_shape']
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(img_shape[0], img_shape[1]),
            mode='bilinear',
            align_corners=False)

        return mask_cls_results, mask_pred_results, all_queries


class SizeList(list):

    def size(self, dim=0):
        if dim > 0:
            return self[0].size(dim - 1)
        return len(self)

    def flatten(self, start_dim=0, end_dim=1):
        # raise NotImplementedError
        if start_dim != 0:
            res = SizeList()
            for i in range(len(self)):
                res.append(self[i].flatten(start_dim - 1, end_dim - 1))
            return res
        elif end_dim > start_dim:  # perform one layer of flatten
            if isinstance(self[0], torch.Tensor):  # for faster speed
                return torch.cat(
                    [i.flatten(0, max(0, end_dim - 1)) for i in self], dim=0)
            res = SizeList()
            for i in range(len(self)):
                res.extend(self[i].flatten(0, end_dim - 1))
            if isinstance(res[0], torch.Tensor):
                return torch.stack(res, dim=0)
            return res
        else:
            return self

    def new_tensor(self, *args, **kwargs):
        return self[0].new_tensor(*args, **kwargs)


@MODELS.register_module()
class QueryAsTrID(MaskFormerFusionHead):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def instance_postprocess(self, mask_cls, mask_pred):
        """Instance segmengation postprocess.

        Args:
            mask_cls (Tensor): Classfication outputs of shape
                (num_queries, cls_out_channels) for a image.
                Note `cls_out_channels` should includes
                background.
            mask_pred (Tensor): Mask outputs of shape
                (num_queries, h, w) for a image.

        Returns:
            tuple[Tensor]: Instance segmentation results.

            - labels_per_image (Tensor): Predicted labels,\
                shape (n, ).
            - bboxes (Tensor): Bboxes and scores with shape (n, 5) of \
                positive region in binary mask, the last column is scores.
            - mask_pred_binary (Tensor): Instance masks of \
                shape (n, h, w).
        """
        max_per_image = self.test_cfg.get('max_per_image', 100)
        num_queries = mask_cls.shape[0]
        # shape (num_queries, num_class)
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        # shape (num_queries * num_class, )
        labels = torch.arange(self.num_classes, device=mask_cls.device).\
            unsqueeze(0).repeat(num_queries, 1).flatten(0, 1)
        scores_per_image, top_indices = scores.flatten(0, 1).topk(
            max_per_image, sorted=False)
        labels_per_image = labels[top_indices]

        # selected query indices in [0, num_queries)
        query_indices = top_indices // self.num_classes
        mask_pred = mask_pred[query_indices]

        # extract things
        is_thing = labels_per_image < self.num_things_classes
        scores_per_image = scores_per_image[is_thing]
        labels_per_image = labels_per_image[is_thing]
        mask_pred = mask_pred[is_thing]

        mask_pred_binary = (mask_pred > 0).float()
        mask_scores_per_image = (mask_pred.sigmoid() *
                                 mask_pred_binary).flatten(1).sum(1) / (
                                     mask_pred_binary.flatten(1).sum(1) + 1e-6)
        det_scores = scores_per_image * mask_scores_per_image
        mask_pred_binary = mask_pred_binary.bool()
        query_indices = query_indices[is_thing]
        bboxes = mask2bbox(mask_pred_binary)
        bboxes = torch.cat([
            query_indices[:, None],
            bboxes,
            det_scores[:, None],
        ],
                           dim=-1)  # Zitong fixed: id needs to be at first

        return labels_per_image, bboxes, mask_pred_binary

    def format_results(self, results):
        """Format the model predictions according to the interface with
        dataset.

        Args:
            results (dict): Processed
                results of single images. Usually contains
                following keys.

                - scores (Tensor): Classification scores, has shape
                  (num_instance,)
                - labels (Tensor): Has shape (num_instances,).
                - masks (Tensor): Processed mask results, has
                  shape (num_instances, h, w).

        Returns:
            tuple: Formatted bbox and mask results.. It contains two items:

                - bbox_results (list[np.ndarray]): BBox results of
                  single image. The list corresponds to each class.
                  each ndarray has a shape (N, 5), N is the number of
                  bboxes with this category, and last dimension
                  5 arrange as (x1, y1, x2, y2, scores).
                - mask_results (list[np.ndarray]): Mask results of
                  single image. The list corresponds to each class.
                  each ndarray has shape (N, img_h, img_w), N
                  is the number of masks with this category.
        """
        data_keys = results.keys()
        # assert 'scores' in data_keys
        assert 'labels' in data_keys

        assert 'masks' in data_keys, \
            'results should contain ' \
            'masks when format the results '
        mask_results = [[] for _ in range(self.num_classes)]

        num_masks = len(results.bboxes)

        if num_masks == 0:
            bbox_results = [
                np.zeros((0, 5), dtype=np.float32)
                for _ in range(self.num_classes)
            ]
            return bbox_results, mask_results

        labels = results.labels.detach().cpu().numpy()

        det_bboxes = results.bboxes
        det_bboxes = det_bboxes.detach().cpu().numpy()
        bbox_results = [
            det_bboxes[labels == i, :] for i in range(self.num_classes)
        ]

        masks = results.masks.detach().cpu().numpy()

        for idx in range(num_masks):
            mask = masks[idx]
            mask_results[labels[idx]].append(mask)

        return bbox_results, mask_results
