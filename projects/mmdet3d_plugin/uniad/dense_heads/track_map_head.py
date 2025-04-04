# Copyright (c) 2025 Yinzhe Shen

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.utils import TORCH_VERSION, digit_version

from mmdet.core import (multi_apply, multi_apply, reduce_mean)
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models import HEADS
from mmdet.models.builder import build_head
from projects.mmdet3d_plugin.uniad.dense_heads import BEVFormerTrackHead, PansegformerHead
from mmdet3d.core.bbox.coders import build_bbox_coder
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
from mmcv.runner import force_fp32, auto_fp16


@HEADS.register_module()
class TrackMapHead(BEVFormerTrackHead):
    """Head of Detr3D.
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
        bev_h, bev_w (int): spatial shape of BEV queries.
    """

    def __init__(self,
                 track_config,
                 map_config,
                 interaction_config,
                 **kwargs):
        super(TrackMapHead, self).__init__(**track_config, **kwargs)
        self.map_head = build_head(map_config)
        self.interaction_head = build_transformer_layer_sequence(interaction_config)
        self.map_pos_encoder = nn.Sequential(
            Linear(4, self.embed_dims),
            nn.ReLU(),
            Linear(self.embed_dims, self.embed_dims),
        )
        assert self.map_head.bev_h == self.bev_h
        assert self.map_head.bev_w == self.bev_w
        assert self.transformer.decoder.num_layers == self.map_head.transformer.decoder.num_layers

    @force_fp32(apply_to=('bev_embed',))
    def get_states_and_refs_joint(self,
                                  bev_embed,
                                  track_query_embeds,
                                  motion_query,
                                  bev_h,
                                  bev_w,
                                  track_ref_points,
                                  track_reg_branches,
                                  track_cls_branches,
                                  img_metas,
                                  ):
        ### Tracking prepare ###
        bs = bev_embed.shape[1]
        track_query_pos, track_query = torch.split(
            track_query_embeds, self.embed_dims, dim=1)
        track_query_pos = track_query_pos.unsqueeze(0).expand(bs, -1, -1)
        track_query = track_query.unsqueeze(0).expand(bs, -1, -1)
        motion_query = motion_query.unsqueeze(0).expand(bs, -1, -1).transpose(0, 1)

        track_reference_points = track_ref_points.unsqueeze(0).expand(bs, -1, -1)
        track_reference_points = track_reference_points.sigmoid()

        track_init_reference_out = track_reference_points
        track_query = track_query.permute(1, 0, 2)
        track_query_pos = track_query_pos.permute(1, 0, 2)

        track_inter_states = []
        track_inter_references = []
        motion_inter_states = []
        ### Tracking prepare ###

        ### Mapping prepare ###
        mlvl_feats = [torch.reshape(bev_embed, (bs, self.map_head.bev_h, self.map_head.bev_w, -1)).permute(0, 3, 1, 2)]
        img_masks = mlvl_feats[0].new_zeros((bs, self.map_head.bev_h, self.map_head.bev_w))

        hw_lvl = [feat_lvl.shape[-2:] for feat_lvl in mlvl_feats]
        mlvl_masks = []
        mlvl_positional_encodings = []
        for feat in mlvl_feats:
            mlvl_masks.append(
                F.interpolate(img_masks[None],
                              size=feat.shape[-2:]).to(torch.bool).squeeze(0))
            mlvl_positional_encodings.append(
                self.map_head.positional_encoding(mlvl_masks[-1]))

        map_query_embeds = None
        if not self.map_head.as_two_stage:
            map_query_embeds = self.map_head.query_embedding.weight

        # map BEV encoding
        (map_query, map_query_pos, memory, lvl_pos_embed_flatten, mask_flatten, map_reference_points, spatial_shapes,
         level_start_index, valid_ratios, init_reference_out, enc_outputs_class, enc_outputs_coord_unact) \
            = self.map_head.transformer.encode(
            mlvl_feats,
            mlvl_masks,
            map_query_embeds,
            mlvl_positional_encodings,
            reg_branches=self.map_head.reg_branches if self.map_head.with_box_refine else None,  # noqa:E501
            cls_branches=self.map_head.cls_branches if self.map_head.as_two_stage else None  # noqa:E501
        )

        map_query = map_query.permute(1, 0, 2)
        memory = memory.permute(1, 0, 2)
        map_query_pos = map_query_pos.permute(1, 0, 2)
        map_inter_states = []
        map_inter_references = []
        ### Mapping prepare ###

        # Joint track and map decoding
        for i in range(self.transformer.decoder.num_layers):
            # tracking query means the active query from the last frame
            # keep them unchanged during interaction encoding
            tracking_query = track_query[900:]
            # interaction encoding
            track_query, map_query = self.interaction_head.forward_single_layer(
                layer_id=i,
                track_query=track_query + track_query_pos,
                map_query=map_query + map_query_pos,
            )
            track_query = torch.cat([track_query[:900], tracking_query], dim=0)

            # motion decoding
            motion_query, _ = self.transformer.motion_decoder.forward_single_layer(
                layer_id=i,
                query=motion_query,
                key=None,
                value=bev_embed,
                query_pos=track_query_pos,
                reference_points=track_reference_points,
                reg_branches=None,
                spatial_shapes=torch.tensor([[bev_h, bev_w]], device=motion_query.device),
                level_start_index=torch.tensor([0], device=motion_query.device),
                img_metas=img_metas
            )

            # track decoding
            track_query, track_reference_points = self.transformer.decoder.forward_single_layer(
                layer_id=i,
                query=track_query,
                key=None,
                value=bev_embed,
                query_pos=track_query_pos,
                reference_points=track_reference_points,
                reg_branches=track_reg_branches,
                spatial_shapes=torch.tensor([[bev_h, bev_w]], device=track_query.device),
                level_start_index=torch.tensor([0], device=track_query.device),
                img_metas=img_metas
            )
            track_query_pos = self.transformer.track_pos_encoder(track_reference_points).permute(1, 0, 2)

            # map decoding
            map_query, map_reference_points = self.map_head.transformer.decoder.forward_single_layer(
                layer_id=i,
                query=map_query,
                key=None,
                value=memory,
                query_pos=map_query_pos,
                key_padding_mask=mask_flatten,
                reference_points=map_reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reg_branches=self.map_head.reg_branches if self.map_head.with_box_refine else None,  # noqa:E501
                )
            map_query_pos = self.map_pos_encoder(map_reference_points).permute(1, 0, 2)

            # store intermediate states
            track_inter_states.append(track_query)
            track_inter_references.append(track_reference_points)
            motion_inter_states.append(motion_query)
            map_inter_states.append(map_query)
            map_inter_references.append(map_reference_points)

        track_inter_states = torch.stack(track_inter_states)
        track_inter_references = torch.stack(track_inter_references)
        motion_inter_states = torch.stack(motion_inter_states)

        map_inter_states = torch.stack(map_inter_states)
        map_inter_references = torch.stack(map_inter_references)

        ### map segmentation decoding ###
        memory = memory
        memory_pos = lvl_pos_embed_flatten
        memory_mask = mask_flatten
        query_pos = map_query_pos
        hs = map_inter_states
        init_reference = init_reference_out
        inter_references = map_inter_references
        if self.map_head.as_two_stage:
            enc_outputs_class = enc_outputs_class
            enc_outputs_coord = enc_outputs_coord_unact
        else:
            enc_outputs_class = None
            enc_outputs_coord = None

        memory = memory.permute(1, 0, 2)
        query = hs[-1].permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        memory_pos = memory_pos.permute(1, 0, 2)

        # we should feed these to mask deocder.
        args_tuple = [memory, memory_mask, memory_pos, query, None, query_pos, hw_lvl]

        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.map_head.cls_branches[lvl](hs[lvl])
            tmp = self.map_head.reg_branches[lvl](hs[lvl])

            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)

        map_outs = {
            'bev_embed': None if self.map_head.as_two_stage else bev_embed,
            'outputs_classes': outputs_classes,
            'outputs_coords': outputs_coords,
            'enc_outputs_class': enc_outputs_class if self.map_head.as_two_stage else None,
            'enc_outputs_coord': enc_outputs_coord.sigmoid() if self.map_head.as_two_stage else None,
            'args_tuple': args_tuple,
            'reference': reference,
        }
        ### Mapping ###

        return track_inter_states, track_init_reference_out, track_inter_references, map_outs, motion_inter_states

    def get_detections_maps(self,
                            bev_embed,
                            object_query_embeds,
                            motion_query,
                            ref_points,
                            img_metas,
                            ):
        assert bev_embed.shape[0] == self.bev_h * self.bev_w

        hs, init_reference, inter_references, map_outs, motion_hs = self.get_states_and_refs_joint(
            bev_embed,
            object_query_embeds,
            motion_query,
            self.bev_h,
            self.bev_w,
            track_ref_points=ref_points,
            track_reg_branches=self.reg_branches if self.with_box_refine else None,
            track_cls_branches=self.cls_branches if self.as_two_stage else None,
            img_metas=img_metas,
        )
        hs = hs.permute(0, 2, 1, 3)
        motion_hs = motion_hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []
        outputs_trajs = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                # reference = init_reference
                reference = ref_points.sigmoid()
                motion_reference = torch.zeros([reference.shape[0], self.past_steps+self.fut_steps, 2], device=reference.device)
            else:
                reference = inter_references[lvl - 1]
                motion_reference = pred_trans
                # ref_size_base = inter_box_sizes[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])  # bbox

            pred_trans = self.past_traj_reg_branches[lvl](motion_hs[lvl]).view(
                -1, self.past_steps + self.fut_steps, 2)
            pred_trans = pred_trans + motion_reference
            pred_traj_past = torch.cumsum(pred_trans[:, :self.past_steps], dim=1)
            pred_traj_past = torch.flip(pred_traj_past, [1]) * -1.0
            pred_traj_fut = torch.cumsum(pred_trans[:, self.past_steps:], dim=1)
            outputs_past_traj = torch.cat([pred_traj_past, pred_traj_fut], dim=1).unsqueeze(0)  # scene centric, past and fut traj

            # estimate velocity from trajectory prediction, future - past / t, on nuscenes t=1s
            velo = outputs_past_traj[0,:,self.past_steps] - outputs_past_traj[0, :, self.past_steps-1]
            # update out dict with new velocity and past trajectory
            tmp[..., -2:] = velo

            # TODO: check the shape of reference
            assert reference.shape[-1] == 3
            tmp[..., 0:2] += reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            tmp[..., 4:5] += reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()

            last_ref_points = torch.cat(
                [tmp[..., 0:2], tmp[..., 4:5]], dim=-1,
            )

            tmp[..., 0:1] = (tmp[..., 0:1] * (self.pc_range[3] -
                                              self.pc_range[0]) + self.pc_range[0])
            tmp[..., 1:2] = (tmp[..., 1:2] * (self.pc_range[4] -
                                              self.pc_range[1]) + self.pc_range[1])
            tmp[..., 4:5] = (tmp[..., 4:5] * (self.pc_range[5] -
                                              self.pc_range[2]) + self.pc_range[2])

            # tmp[..., 2:4] = tmp[..., 2:4] + ref_size_basse[..., 0:2]
            # tmp[..., 5:6] = tmp[..., 5:6] + ref_size_basse[..., 2:3]

            # TODO: check if using sigmoid
            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            outputs_trajs.append(outputs_past_traj)
        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        outputs_trajs = torch.stack(outputs_trajs)
        last_ref_points = inverse_sigmoid(last_ref_points)
        det_outs = {
            'all_cls_scores': outputs_classes,
            'all_bbox_preds': outputs_coords,
            'all_past_traj_preds': outputs_trajs,
            'enc_cls_scores': None,
            'enc_bbox_preds': None,
            'last_ref_points': last_ref_points,
            'query_feats': hs,
            'motion_query_feats': motion_hs,
        }
        return det_outs, map_outs
