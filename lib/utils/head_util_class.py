import tensorflow as tf
import numpy as np
import utils.tf_util as tf_util
import dataset.maps_dict as maps_dict

from functools import partial
from core.config import cfg


def select_conv_op(op_type):
    if op_type == 'conv1d':  # conv1d function
        conv_op = partial(
            tf_util.conv1d,
            kernel_size=1,
            padding='VALID',
        )
    elif op_type == 'conv2d':
        conv_op = partial(
            tf_util.conv2d,
            kernel_size=[1, 1],
            padding='VALID',
        )
    elif op_type == 'fc':
        conv_op = tf_util.fully_connected
    return conv_op


class box_regression_head(tf.keras.Model):
    def __init__(self, pred_cls_channel, pred_reg_base_num, pred_reg_channel_num, bn, pred_attr_velo, conv_op):
        super(box_regression_head, self).__init__()
        self.pred_cls_channel = pred_cls_channel
        self.pred_reg_base_num = pred_reg_base_num
        self.pred_reg_channel_num = pred_reg_channel_num
        self.bn = bn
        self.pred_attr_velo = pred_attr_velo
        if conv_op != "conv1d":
            raise ("conv1d is not implemented")
        if pred_attr_velo:
            raise ("pred_attr_velo is not implemented")

        self.class_layer = []
        self.class_layer.append(
            tf.keras.layers.Conv1D(128, 1, padding='valid', activation=tf.nn.relu, name="pred_cls_base"))
        if self.bn is True:
            if cfg.MODEL.NETWORK.USE_GN:
                raise ("Group Normalization is not implemented.")
            else:
                self.class_layer.append(
                    tf.keras.layers.BatchNormalization(name="pred_cls_base"))
        self.class_layer.append(
            tf.keras.layers.Conv1D(pred_cls_channel, 1, padding='valid', activation=None, name="pred_cls"))

        self.bbox_layer = []
        self.bbox_layer.append(
            tf.keras.layers.Conv1D(128, 1, padding='valid', activation=tf.nn.relu, name="pred_reg_base"))
        if self.bn is True:
            if cfg.MODEL.NETWORK.USE_GN:
                raise ("Group Normalization is not implemented.")
            else:
                self.bbox_layer.append(
                    tf.keras.layers.BatchNormalization(name="pred_reg_base"))
        self.bbox_layer.append(
            tf.keras.layers.Conv1D(pred_reg_base_num *
                                   (pred_reg_channel_num + cfg.MODEL.ANGLE_CLS_NUM * 2), 1, padding='valid', activation=None, name="pred_reg"))

    def call(self, feature_input, is_training, bn_decay, output_dict):
        """
        Construct box-regression head
        """
        bs, points_num, _ = feature_input.get_shape().as_list()
        # classification
        pred_cls = self.class_layer[0](feature_input, training=is_training)
        for i in range(1, len(self.class_layer)):
            # update bn_decay(It may have a bug. momentum = bn_decay)
            if hasattr(self.class_layer[i], 'momentum') and bn_decay is not None:
                self.class_layer[i].momentum = (1-bn_decay)

            pred_cls = self.class_layer[i](
                pred_cls, training=is_training)
        # pred_cls = conv_op(feature_input, 128, scope='pred_cls_base',
        #                    bn=bn, is_training=is_training, bn_decay=bn_decay)
        # pred_cls = conv_op(pred_cls, pred_cls_channel,
        #                    activation_fn=None, scope='pred_cls')

        # bounding-box prediction
        # pred_reg = conv_op(feature_input, 128, bn=bn, is_training=is_training,
        #                    scope='pred_reg_base', bn_decay=bn_decay)
        # pred_reg = conv_op(pred_reg, pred_reg_base_num * (pred_reg_channel_num +
        #                                                   cfg.MODEL.ANGLE_CLS_NUM * 2), activation_fn=None, scope='pred_reg')
        pred_reg = self.bbox_layer[0](feature_input, training=is_training)
        for i in range(1, len(self.bbox_layer)):
            # update bn_decay(It may have a bug. momentum = bn_decay)
            if hasattr(self.bbox_layer[i], 'momentum') and bn_decay is not None:
                self.bbox_layer[i].momentum = (1-bn_decay)

            pred_reg = self.bbox_layer[i](
                pred_reg, training=is_training)
        pred_reg = tf.reshape(pred_reg, [
            bs, points_num, self.pred_reg_base_num, self.pred_reg_channel_num + cfg.MODEL.ANGLE_CLS_NUM * 2])

        # if pred_attr_velo:  # velocity and attribute
        #     pred_attr = conv_op(feature_input, 128, bn=bn, is_training=is_training,
        #                         scope='pred_attr_base', bn_decay=bn_decay)
        #     pred_attr = conv_op(pred_attr, pred_reg_base_num * 8,
        #                         activation_fn=None, scope='pred_attr')
        #     pred_attr = tf.reshape(
        #         pred_attr, [bs, points_num, pred_reg_base_num, 8])

        #     pred_velo = conv_op(feature_input, 128, bn=bn, is_training=is_training,
        #                         scope='pred_velo_base', bn_decay=bn_decay)
        #     pred_velo = conv_op(pred_velo, pred_reg_base_num * 2,
        #                         activation_fn=None, scope='pred_velo')
        #     pred_velo = tf.reshape(
        #         pred_velo, [bs, points_num, pred_reg_base_num, 2])

        #     output_dict[maps_dict.PRED_ATTRIBUTE].append(pred_attr)
        #     output_dict[maps_dict.PRED_VELOCITY].append(pred_velo)

        output_dict[maps_dict.PRED_CLS].append(pred_cls)
        output_dict[maps_dict.PRED_OFFSET].append(
            tf.slice(pred_reg, [0, 0, 0, 0], [-1, -1, -1, self.pred_reg_channel_num]))
        output_dict[maps_dict.PRED_ANGLE_CLS].append(tf.slice(pred_reg,
                                                              [0, 0, 0, self.pred_reg_channel_num], [-1, -1, -1, cfg.MODEL.ANGLE_CLS_NUM]))
        output_dict[maps_dict.PRED_ANGLE_RES].append(tf.slice(pred_reg,
                                                              [0, 0, 0, self.pred_reg_channel_num+cfg.MODEL.ANGLE_CLS_NUM], [-1, -1, -1, -1]))

        return


class iou_regression_head(tf.keras.Model):
    def __init__(self, pred_cls_channel, bn, conv_op):
        super(iou_regression_head, self).__init__()
        self.pred_cls_channel = pred_cls_channel
        self.bn = bn
        if conv_op != "conv1d":
            raise ("conv1d is not implemented")

        self.layer_list = []
        self.layer_list.append(
            tf.keras.layers.Conv1D(128, 1, padding='valid', activation=tf.nn.relu))
        if self.bn is True:
            if cfg.MODEL.NETWORK.USE_GN:
                raise ("Group Normalization is not implemented.")
            else:
                self.layer_list.append(
                    tf.keras.layers.BatchNormalization())
        self.layer_list.append(
            tf.keras.layers.Conv1D(pred_cls_channel, 1, padding='valid', activation=tf.nn.relu))

    def iou_regression_head(self, feature_input,  is_training, bn_decay, output_dict):
        """
        Construct iou-prediction head:
        """
        bs, points_num, _ = feature_input.get_shape().as_list()
        # classification
        pred_iou = self.layer_list[0](feature_input, training=is_training)
        for i in range(1, len(self.layer_list)):
            # update bn_decay(It may have a bug. momentum = bn_decay)
            if hasattr(self.layer_list[i], 'momentum') and bn_decay is not None:
                self.layer_list[i].momentum = (1-bn_decay)

            pred_iou = self.layer_list[i](
                pred_iou, training=is_training)
        # pred_iou = conv_op(feature_input, 128, scope='pred_iou_base',
        #                    bn=bn, is_training=is_training, bn_decay=bn_decay)
        # pred_iou = conv_op(pred_iou, pred_cls_channel,
        #                    activation_fn=None, scope='pred_iou')

        output_dict[maps_dict.PRED_IOU_3D_VALUE].append(pred_iou)

        return
