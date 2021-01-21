import tensorflow as tf
import numpy as np

import utils.tf_util as tf_util

from core.config import cfg
from utils.layers_util import *

import dataset.maps_dict as maps_dict


class LayerBuilder(tf.keras.Model):
    def __init__(self, layer_idx, is_training, layer_cfg):
        super(LayerBuilder, self).__init__()
        self.layer_idx = layer_idx
        self.is_training = is_training

        self.layer_architecture = layer_cfg[self.layer_idx]

        self.xyz_index = self.layer_architecture[0]
        self.feature_index = self.layer_architecture[1]
        self.radius_list = self.layer_architecture[2]
        self.nsample_list = self.layer_architecture[3]
        self.mlp_list = self.layer_architecture[4]
        self.bn = self.layer_architecture[5]

        self.fps_sample_range_list = self.layer_architecture[6]
        self.fps_method_list = self.layer_architecture[7]
        self.npoint_list = self.layer_architecture[8]
        assert len(self.fps_sample_range_list) == len(self.fps_method_list)
        assert len(self.fps_method_list) == len(self.npoint_list)

        self.former_fps_idx = self.layer_architecture[9]
        self.use_attention = self.layer_architecture[10]
        self.layer_type = self.layer_architecture[11]
        self.scope = self.layer_architecture[12]
        self.dilated_group = self.layer_architecture[13]
        self.vote_ctr_index = self.layer_architecture[14]
        self.aggregation_channel = self.layer_architecture[15]

        if self.layer_type in ['SA_Layer', 'Vote_Layer', 'SA_Layer_SSG_Last']:
            assert len(self.xyz_index) == 1
        elif self.layer_type == 'FP_Layer':
            assert len(self.xyz_index) == 2
        else:
            raise Exception('Not Implementation Error!!!')

        self.layer_list = []
        if self.layer_type == "SA_Layer":
            for i in range(len(self.mlp_list)):
                tmp_layer_list = []
                for j in range(len(self.mlp_list[i])):
                    tmp_layer_list.append(
                        tf.keras.layers.Conv2D(
                            self.mlp_list[i][j], kernel_size=(1, 1),
                            strides=(1, 1), padding='valid', activation=tf.nn.relu, name="layer{}/conv{}_{}".format(layer_idx+1, i, j)))
                    if self.bn is True:
                        if cfg.MODEL.NETWORK.USE_GN:
                            raise ("Group Normalization is not implemented.")
                        else:
                            tmp_layer_list.append(
                                tf.keras.layers.BatchNormalization(name="layer{}/conv{}_{}".format(layer_idx+1, i, j)))
                self.layer_list.append(tmp_layer_list)

            if cfg.MODEL.NETWORK.AGGREGATION_SA_FEATURE:
                self.aggregation_layer = []
                self.aggregation_layer.append(tf.keras.layers.Conv1D(
                    self.aggregation_channel, 1, padding='valid', name="layer{}/ensemble".format(layer_idx+1)))
                if cfg.MODEL.NETWORK.USE_GN:
                    raise ("Group Normalization is not implemented.")
                else:
                    self.aggregation_layer.append(
                        tf.keras.layers.BatchNormalization(name="layer{}/ensemble".format(layer_idx+1)))

        elif self.layer_type == "Vote_Layer":
            for i, channel in enumerate(self.mlp_list):
                self.layer_list.append(tf.keras.layers.Conv1D(
                    channel, 1, padding='valid', activation=tf.nn.relu, name="vote/vote_layer{}".format(i)))
                if self.bn is True:
                    if cfg.MODEL.NETWORK.USE_GN:
                        raise ("Group Normalization is not implemented.")
                    else:
                        self.layer_list.append(
                            tf.keras.layers.BatchNormalization(name="vote/vote_layer{}".format(i)))

            self.layer_list.append(tf.keras.layers.Conv1D(
                3, 1, padding='valid', activation=None, name="vote/vote_offsets"))
        else:
            raise("{} is not implemented.".format(self.layer_type))

        # else:
        #     raise ("{} is not implemented.".format(self.layer_type))

    def build_layer(self, xyz_list, feature_list, fps_idx_list, bn_decay, output_dict):
        """
        Build layers
        """

        xyz_input = []
        for xyz_index in self.xyz_index:
            xyz_input.append(xyz_list[xyz_index])

        feature_input = []
        for feature_index in self.feature_index:
            feature_input.append(feature_list[feature_index])

        if self.former_fps_idx != -1:
            former_fps_idx = fps_idx_list[self.former_fps_idx]
        else:
            former_fps_idx = None

        if self.vote_ctr_index != -1:
            vote_ctr = xyz_list[self.vote_ctr_index]
        else:
            vote_ctr = None

        if self.layer_type == 'SA_Layer':
            new_xyz, new_points, new_fps_idx = self.pointnet_sa_module_msg(
                xyz_input[0], feature_input[0],
                self.radius_list, self.nsample_list,
                self.mlp_list, self.is_training, bn_decay, self.bn,
                self.fps_sample_range_list, self.fps_method_list, self.npoint_list,
                former_fps_idx, self.use_attention, self.scope,
                self.dilated_group, vote_ctr, self.aggregation_channel)
            xyz_list.append(new_xyz)
            feature_list.append(new_points)
            fps_idx_list.append(new_fps_idx)

        elif self.layer_type == 'SA_Layer_SSG_Last':
            new_points = pointnet_sa_module(
                xyz_input[0], feature_input[0],
                self.mlp_list, self.is_training, bn_decay,
                self.bn, self.scope,
            )
            xyz_list.append(None)
            feature_list.append(new_points)
            fps_idx_list.append(None)

        elif self.layer_type == 'FP_Layer':
            new_points = pointnet_fp_module(xyz_input[0], xyz_input[1], feature_input[0],
                                            feature_input[1], self.mlp_list, self.is_training, bn_decay, self.scope, self.bn)
            xyz_list.append(xyz_input[0])
            feature_list.append(new_points)
            fps_idx_list.append(None)

        elif self.layer_type == 'Vote_Layer':
            new_xyz, new_points, ctr_offsets = self.vote_layer(
                xyz_input[0], feature_input[0], self.mlp_list, self.is_training, bn_decay, self.bn, self.scope)
            output_dict[maps_dict.PRED_VOTE_BASE].append(xyz_input[0])
            output_dict[maps_dict.PRED_VOTE_OFFSET].append(ctr_offsets)
            xyz_list.append(new_xyz)
            feature_list.append(new_points)
            fps_idx_list.append(None)

        return xyz_list, feature_list, fps_idx_list

    def pointnet_sa_module_msg(self, xyz, points, radius_list, nsample_list,
                               mlp_list, is_training, bn_decay, bn,
                               fps_sample_range_list, fps_method_list, npoint_list,
                               former_fps_idx, use_attention, scope,
                               dilated_group, vote_ctr=None, aggregation_channel=None,
                               debugging=False,
                               epsilon=1e-5):
        ''' PointNet Set Abstraction (SA) module with Multi-Scale Grouping (MSG)
            Input:
                xyz: (batch_size, ndataset, 3) TF tensor
                points: (batch_size, ndataset, channel) TF tensor
                npoint: int -- points sampled in farthest point sampling
                radius_list: list of float32 -- search radius in local region
                nsample_list: list of int32 -- how many points in each local region
                mlp_list: list of list of int32 -- output size for MLP on each point
                fps_method: 'F-FPS', 'D-FPS', 'FS'
                fps_start_idx: 
            Return:
                new_xyz: (batch_size, npoint, 3) TF tensor
                new_points: (batch_size, npoint, \sum_k{mlp[k][-1]}) TF tensor
        '''
        bs = xyz.get_shape().as_list()[0]
        # with tf.variable_scope(scope) as sc:
        cur_fps_idx_list = []
        last_fps_end_index = 0
        for fps_sample_range, fps_method, npoint in zip(fps_sample_range_list, fps_method_list, npoint_list):
            tmp_xyz = tf.slice(
                xyz, [0, last_fps_end_index, 0], [-1, fps_sample_range, -1])
            tmp_points = tf.slice(
                points, [0, last_fps_end_index, 0], [-1, fps_sample_range, -1])
            if npoint == 0:
                last_fps_end_index += fps_sample_range
                continue
            if vote_ctr is not None:
                npoint = vote_ctr.get_shape().as_list()[1]
                fps_idx = tf.tile(tf.reshape(
                    tf.range(npoint), [1, npoint]), [bs, 1])
            elif fps_method == 'FS':
                features_for_fps = tf.concat(
                    [tmp_xyz, tmp_points], axis=-1)
                features_for_fps_distance = model_util.calc_square_dist(
                    features_for_fps, features_for_fps, norm=False)
                fps_idx_1 = farthest_point_sample_with_distance(
                    npoint, features_for_fps_distance)
                fps_idx_2 = farthest_point_sample(npoint, tmp_xyz)
                # [bs, npoint * 2]
                fps_idx = tf.concat([fps_idx_1, fps_idx_2], axis=-1)
            elif npoint == tmp_xyz.get_shape().as_list()[1]:
                fps_idx = tf.tile(tf.reshape(
                    tf.range(npoint), [1, npoint]), [bs, 1])
            elif fps_method == 'F-FPS':
                features_for_fps = tf.concat(
                    [tmp_xyz, tmp_points], axis=-1)
                features_for_fps_distance = model_util.calc_square_dist(
                    features_for_fps, features_for_fps, norm=False)
                fps_idx = farthest_point_sample_with_distance(
                    npoint, features_for_fps_distance)
            else:  # D-FPS
                fps_idx = farthest_point_sample(npoint, tmp_xyz)

            fps_idx = fps_idx + last_fps_end_index
            cur_fps_idx_list.append(fps_idx)
            last_fps_end_index += fps_sample_range
        fps_idx = tf.concat(cur_fps_idx_list, axis=-1)

        if former_fps_idx is not None:
            fps_idx = tf.concat([fps_idx, former_fps_idx], axis=-1)

        if vote_ctr is not None:
            new_xyz = gather_point(vote_ctr, fps_idx)
        else:
            new_xyz = gather_point(xyz, fps_idx)

        # if deformed_xyz is not None, then no attention model
        if use_attention:
            # first gather the points out
            new_points = gather_point(points, fps_idx)  # [bs, npoint, c]

            # choose farthest feature to center points
            # [bs, npoint, ndataset]
            relation = model_util.calc_square_dist(new_points, points)
            # choose these points with largest distance to center_points
            _, relation_idx = tf.nn.top_k(
                relation, k=relation.shape.as_list()[-1])

        idx_list, pts_cnt_list = [], []
        cur_radius_list = []
        for i in range(len(radius_list)):
            radius = radius_list[i]
            nsample = nsample_list[i]
            if dilated_group:
                # cfg.POINTNET.DILATED_GROUPING
                if i == 0:
                    min_radius = 0.
                else:
                    min_radius = radius_list[i - 1]
                idx, pts_cnt = query_ball_point_dilated(
                    min_radius, radius, nsample, xyz, new_xyz)
            elif use_attention:
                idx, pts_cnt = query_ball_point_withidx(
                    radius, nsample, xyz, new_xyz, relation_idx)
            else:
                idx, pts_cnt = query_ball_point(
                    radius, nsample, xyz, new_xyz)
            idx_list.append(idx)
            pts_cnt_list.append(pts_cnt)

        # debugging
        debugging_list = []
        new_points_list = []
        for i in range(len(radius_list)):
            nsample = nsample_list[i]
            idx, pts_cnt = idx_list[i], pts_cnt_list[i]
            radius = radius_list[i]

            pts_cnt_mask = tf.cast(tf.greater(
                pts_cnt, 0), tf.int32)  # [bs, npoint]
            pts_cnt_fmask = tf.cast(pts_cnt_mask, tf.float32)
            # [bs, npoint, nsample]
            idx = idx * tf.expand_dims(pts_cnt_mask, axis=2)
            grouped_xyz = group_point(xyz, idx)
            original_xyz = grouped_xyz
            grouped_xyz -= tf.expand_dims(new_xyz, 2)
            grouped_points = group_point(points, idx)

            grouped_points = tf.concat(
                [grouped_points, grouped_xyz], axis=-1)

            # for j, num_out_channel in enumerate(mlp_list[i]):
            #     grouped_points = tf_util.conv2d(grouped_points,
            #                                     num_out_channel,
            #                                     [1, 1],
            #                                     padding='VALID',
            #                                     stride=[1, 1],
            #                                     bn=bn,
            #                                     is_training=is_training,
            #                                     scope='conv%d_%d' % (i, j),
            #                                     bn_decay=bn_decay)
            # grouped_points: (4, 4096, 32, 4)
            for j, num_out_channel in enumerate(self.layer_list[i]):
                if hasattr(self.layer_list[i][j], 'momentum') and bn_decay is not None:
                    self.layer_list[i][j].momentum = (1-bn_decay)
                grouped_points = self.layer_list[i][j](
                    grouped_points, training=is_training)

            new_points = tf.reduce_max(grouped_points, axis=[2])

            new_points *= tf.expand_dims(pts_cnt_fmask, axis=-1)
            new_points_list.append(new_points)

        if len(new_points_list) > 0:
            new_points_concat = tf.concat(new_points_list, axis=-1)
            if cfg.MODEL.NETWORK.AGGREGATION_SA_FEATURE:
                for i in range(len(self.aggregation_layer)):
                    # update bn_decay(It may have a bug. momentum = bn_decay)
                    if hasattr(self.aggregation_layer[i], 'momentum') and bn_decay is not None:
                        self.aggregation_layer[i].momentum = (1-bn_decay)

                    new_points_concat = self.aggregation_layer[i](
                        new_points_concat, training=is_training)
                # new_points_concat = tf_util.conv1d(
                #     new_points_concat, aggregation_channel, 1, padding='VALID', bn=bn, is_training=is_training, scope='ensemble', bn_decay=bn_decay)
        else:
            new_points_concat = gather_point(points, fps_idx)

        return new_xyz, new_points_concat, fps_idx

    def vote_layer(self, xyz, points, mlp_list, is_training, bn_decay, bn, scope):
        """
        Voting layer
        """
        # for i, channel in enumerate(mlp_list):
        #     points = tf_util.conv1d(points, channel, 1, padding='VALID', stride=1, bn=bn,
        #                             scope='vote_layer_%d' % i, bn_decay=bn_decay, is_training=is_training)
        # ctr_offsets = tf_util.conv1d(
        #     points, 3, 1, padding='VALID', stride=1, bn=False, activation_fn=None, scope='vote_offsets')

        for i in range(len(self.layer_list)-1):
            # update bn_decay(It may have a bug. momentum = bn_decay)
            if hasattr(self.layer_list[i], 'momentum') and bn_decay is not None:
                self.layer_list[i].momentum = (1-bn_decay)
            points = self.layer_list[i](points, training=is_training)

        ctr_offsets = self.layer_list[-1](points, training=is_training)

        min_offset = tf.reshape(cfg.MODEL.MAX_TRANSLATE_RANGE, [1, 1, 3])
        limited_ctr_offsets = tf.minimum(
            tf.maximum(ctr_offsets, min_offset), -min_offset)
        xyz = xyz + limited_ctr_offsets

        return xyz, points, ctr_offsets
