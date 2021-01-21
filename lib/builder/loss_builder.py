import tensorflow as tf
import numpy as np

from core.config import cfg
from utils.box_3d_utils import transfer_box3d_to_corners
from utils.tf_ops.sampling.tf_sampling import gather_point
from utils.tf_ops.evaluation.tf_evaluate import calc_iou_match_warper
from utils.rotation_util import rotate_points
from np_functions.gt_sampler import vote_targets_np

import utils.model_util as model_util
import dataset.maps_dict as maps_dict


class LossBuilder:
    def __init__(self, stage):
        if stage == 0:
            self.loss_cfg = cfg.MODEL.FIRST_STAGE
        elif stage == 1:
            self.loss_cfg = cfg.MODEL.SECOND_STAGE

        self.stage = stage

        self.cls_loss_type = self.loss_cfg.CLASSIFICATION_LOSS.TYPE
        self.ctr_ness_range = self.loss_cfg.CLASSIFICATION_LOSS.CENTER_NESS_LABEL_RANGE
        self.cls_activation = self.loss_cfg.CLS_ACTIVATION

        if self.cls_loss_type == 'Center-ness' or self.cls_loss_type == 'Focal-loss':
            assert self.cls_activation == 'Sigmoid'

        self.cls_list = cfg.DATASET.KITTI.CLS_LIST

        reg_cfg = self.loss_cfg.REGRESSION_METHOD
        self.reg_type = reg_cfg.TYPE
        self.reg_bin_cls_num = reg_cfg.BIN_CLASS_NUM
        if self.reg_type == 'Bin-Anchor':
            self.offset_loss = self.offset_loss_bin
        else:
            self.offset_loss = self.offset_loss_res

    # def forward(self, index, label_dict, pred_dict, placeholders, corner_loss=False, vote_loss=False, attr_velo_loss=False, iou_loss=False):
    def forward(self, index, label_dict, pred_dict, data, corner_loss=False, vote_loss=False, attr_velo_loss=False, iou_loss=False):
        loss = {}
        loss["cls_loss"] = self.cls_loss(index, label_dict, pred_dict)
        loss["offset_loss"] = self.offset_loss(index, label_dict, pred_dict)
        # loss["angle_loss"] = self.angle_loss(index, label_dict, pred_dict)
        loss["bin_loss"], loss["res_loss"] = self.angle_loss(
            index, label_dict, pred_dict)
        if corner_loss:
            loss["corner_loss"] = self.corner_loss(
                index, label_dict, pred_dict)
        if vote_loss:
            loss["vote_loss"] = self.vote_loss(index, pred_dict, data)
        if attr_velo_loss:
            loss["velo_attr_loss"] = self.velo_attr_loss(
                index, label_dict, pred_dict)
        if iou_loss:
            loss["iou_loss"] = self.iou_loss(index, label_dict, pred_dict)
        # with tf.name_scope('loss_stage%d' % self.stage) as scope:
        #     self.cls_loss(index, label_dict, pred_dict)
        #     self.offset_loss(index, label_dict, pred_dict)
        #     self.angle_loss(index, label_dict, pred_dict)

        #     if corner_loss:
        #         self.corner_loss(index, label_dict, pred_dict)
        #     if vote_loss:
        #         self.vote_loss(index, pred_dict, placeholders)
        #     if attr_velo_loss:
        #         self.velo_attr_loss(index, label_dict, pred_dict)
        #     if iou_loss:
        #         self.iou_loss(index, label_dict, pred_dict)

        return loss

    def cls_loss(self, index, label_dict, pred_dict):
        pmask = label_dict[maps_dict.GT_PMASK][index]  # bs, pts_num, cls_num
        nmask = label_dict[maps_dict.GT_NMASK][index]
        gt_cls = label_dict[maps_dict.GT_CLS][index]  # bs, pts_num

        cls_mask = pmask + nmask
        cls_mask = tf.reduce_max(cls_mask, axis=-1)  # [bs, pts_num]

        pred_cls = pred_dict[maps_dict.PRED_CLS][index]  # bs, pts_num, c

        norm_param = tf.maximum(1., tf.reduce_sum(cls_mask))

        if self.cls_activation == 'Sigmoid':
            gt_cls = tf.cast(tf.one_hot(
                gt_cls - 1, depth=len(self.cls_list), on_value=1, off_value=0, axis=-1), tf.float32)

        if self.cls_loss_type == 'Is-Not':  # Is or Not
            if self.cls_activation == 'Softmax':
                cls_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=gt_cls, logits=pred_cls)
            else:
                cls_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=gt_cls, logits=pred_cls)
                cls_loss = tf.reduce_mean(cls_loss, axis=-1)

        elif self.cls_loss_type == 'Focal-loss':  # Focal-loss producer
            cls_loss = model_util.focal_loss_producer(pred_cls, gt_cls)
            cls_loss = tf.reduce_mean(cls_loss, axis=-1)

        elif self.cls_loss_type == 'Center-ness':  # Center-ness label
            base_xyz = pred_dict[maps_dict.KEY_OUTPUT_XYZ][index]
            base_xyz = tf.stop_gradient(base_xyz)
            assigned_boxes_3d = label_dict[maps_dict.GT_BOXES_ANCHORS_3D][index]
            ctr_ness = self._generate_centerness_label(
                base_xyz, assigned_boxes_3d, pmask)
            gt_cls = gt_cls * tf.expand_dims(ctr_ness, axis=-1)
            cls_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=gt_cls, logits=pred_cls)
            cls_loss = tf.reduce_mean(cls_loss, axis=-1)

        cls_loss = tf.reduce_sum(cls_loss * cls_mask) / norm_param
        cls_loss = tf.identity(cls_loss, 'cls_loss%d' % index)
        tf.summary.scalar('cls_loss%d' % index, cls_loss)
        # tf.add_to_collection(tf.GraphKeys.LOSSES, cls_loss)
        return cls_loss

    def _generate_centerness_label(self, base_xyz, assigned_boxes_3d, pmask, epsilon=1e-6):
        """
        base_xyz: [bs, pts_num, 3]
        assigned_boxes_3d: [bs, pts_num, cls_num, 7]
        pmask: [bs, pts_num, cls_num]

        return: [bs, pts_num]
        """
        bs, pts_num, _ = base_xyz.get_shape().as_list()

        # [bs, pts_num, 7]
        assigned_boxes_3d = tf.reduce_sum(
            assigned_boxes_3d * tf.expand_dims(pmask, axis=-1), axis=2)
        pmask = tf.reduce_max(pmask, axis=2)  # [bs, pts_num]

        canonical_xyz = base_xyz - assigned_boxes_3d[:, :, :3]
        canonical_xyz = tf.reshape(canonical_xyz, [bs * pts_num, 1, 3])
        rys = tf.reshape(assigned_boxes_3d[:, :, -1], [bs * pts_num])
        canonical_xyz = rotate_points(canonical_xyz, -rys)
        canonical_xyz = tf.reshape(canonical_xyz, [bs, pts_num, 3])

        distance_front = assigned_boxes_3d[:,
                                           :, 3] / 2. - canonical_xyz[:, :, 0]
        distance_back = canonical_xyz[:, :, 0] + \
            assigned_boxes_3d[:, :, 3] / 2.
        distance_bottom = 0 - canonical_xyz[:, :, 1]
        distance_top = canonical_xyz[:, :, 1] + assigned_boxes_3d[:, :, 4]
        distance_left = assigned_boxes_3d[:,
                                          :, 5] / 2. - canonical_xyz[:, :, 2]
        distance_right = canonical_xyz[:, :, 2] + \
            assigned_boxes_3d[:, :, 5] / 2.

        ctr_ness_l = tf.minimum(distance_front, distance_back) / \
            tf.maximum(distance_front, distance_back) * pmask
        ctr_ness_w = tf.minimum(distance_left, distance_right) / \
            tf.maximum(distance_left, distance_right) * pmask
        ctr_ness_h = tf.minimum(distance_bottom, distance_top) / \
            tf.maximum(distance_bottom, distance_top) * pmask
        ctr_ness = tf.maximum(ctr_ness_l * ctr_ness_h * ctr_ness_w, epsilon)
        ctr_ness = tf.pow(ctr_ness, 1/3.)  # [bs, points_num]

        min_ctr_ness, max_ctr_ness = self.ctr_ness_range
        ctr_ness_range = max_ctr_ness - min_ctr_ness
        ctr_ness *= ctr_ness_range
        ctr_ness += min_ctr_ness

        return ctr_ness

    def iou_loss(self, index, label_dict, pred_dict):
        pmask = label_dict[maps_dict.GT_PMASK][index]
        pmask = tf.reduce_max(pmask, axis=-1)

        gt_cls = label_dict[maps_dict.GT_CLS][index]  # bs, pts_num
        gt_cls = tf.cast(tf.one_hot(gt_cls - 1, depth=len(self.cls_list), on_value=1,
                                    off_value=0, axis=-1), tf.float32)  # [bs, pts_num, cls_num]

        assigned_gt_boxes_3d = label_dict[maps_dict.GT_BOXES_ANCHORS_3D][index]
        proposals = pred_dict[maps_dict.KEY_ANCHORS_3D][index]

        # bs, proposal_num, cls_num
        target_iou_bev, target_iou_3d = calc_iou_match_warper(
            proposals, assigned_gt_boxes_3d)
        # then normalize target_iou to [-1, 1]
        target_iou_3d = target_iou_3d * 2. - 1.
        target_iou_3d = target_iou_3d * gt_cls

        pred_iou = pred_dict[maps_dict.PRED_IOU_3D_VALUE][index]

        norm_param = tf.maximum(1., tf.reduce_sum(pmask))

        iou_loss = model_util.huber_loss(pred_iou - target_iou_3d, delta=1.)
        iou_loss = tf.reduce_mean(iou_loss, axis=-1) * pmask
        iou_loss = tf.identity(tf.reduce_sum(
            iou_loss) / norm_param, 'iou_loss%d' % index)
        tf.summary.scalar('iou_loss%d' % index, iou_loss)
        # tf.add_to_collection(tf.GraphKeys.LOSSES, iou_loss)
        return iou_loss

    # def vote_loss(self, index, pred_dict, placeholders):

    def vote_loss(self, index, pred_dict, data):
        vote_offset = pred_dict[maps_dict.PRED_VOTE_OFFSET][index]
        vote_base = pred_dict[maps_dict.PRED_VOTE_BASE][index]
        bs, pts_num, _ = vote_offset.get_shape().as_list()
        # gt_boxes_3d = placeholders[maps_dict.PL_LABEL_BOXES_3D]
        gt_boxes_3d = data[maps_dict.PL_LABEL_BOXES_3D]
        # vote_mask, vote_target = tf.py_func(
        #     vote_targets_np, [vote_base, gt_boxes_3d], [tf.float32, tf.float32])
        vote_mask, vote_target = tf.compat.v1.py_func(
            vote_targets_np, [vote_base, gt_boxes_3d], [tf.float32, tf.float32])
        vote_mask = tf.reshape(vote_mask, [bs, pts_num])
        vote_target = tf.reshape(vote_target, [bs, pts_num, 3])

        vote_loss = tf.reduce_sum(model_util.huber_loss(
            vote_target - vote_offset, delta=1.), axis=-1) * vote_mask
        vote_loss = tf.reduce_sum(vote_loss) / \
            tf.maximum(1., tf.reduce_sum(vote_mask))
        vote_loss = tf.identity(vote_loss, 'vote_loss%d' % index)
        tf.summary.scalar('vote_loss%d' % index, vote_loss)
        # tf.add_to_collection(tf.GraphKeys.LOSSES, vote_loss)
        return vote_loss

    def velo_attr_loss(self, index, label_dict, pred_dict):
        pmask = label_dict[maps_dict.GT_PMASK][index]
        nmask = label_dict[maps_dict.GT_NMASK][index]
        # bs, pts_num, cls_num
        gt_attribute = label_dict[maps_dict.GT_ATTRIBUTE][index]
        # bs,pts_num,cls_num,2
        gt_velocity = label_dict[maps_dict.GT_VELOCITY][index]

        pred_attribute = pred_dict[maps_dict.PRED_ATTRIBUTE][index]
        pred_velocity = pred_dict[maps_dict.PRED_VELOCITY][index]

        attr_mask = tf.cast(tf.greater_equal(gt_attribute, 0), tf.float32)
        attr_mask = attr_mask * pmask
        gt_attribute_onehot = tf.cast(tf.one_hot(
            gt_attribute, depth=8, on_value=1, off_value=0, axis=-1), tf.float32)  # [bs, pts_num, cls_num, 8]
        attr_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=gt_attribute_onehot, logits=pred_attribute)
        attr_loss = attr_loss * tf.expand_dims(attr_mask, axis=-1)
        attr_loss = tf.reduce_sum(
            attr_loss) / (tf.maximum(1., tf.reduce_sum(attr_mask)) * 8.)
        attr_loss = tf.identity(attr_loss, 'attribute_loss_%d' % index)
        tf.summary.scalar('attribute_loss_%d' % index, attr_loss)
        tf.add_to_collection(tf.GraphKeys.LOSSES, attr_loss)

        velo_mask = tf.cast(tf.logical_not(
            tf.is_nan(tf.reduce_sum(gt_velocity, axis=-1))), tf.float32)
        velo_mask = velo_mask * pmask
        zero_velocity = tf.zeros_like(gt_velocity)
        gt_velocity = tf.where(tf.is_nan(gt_velocity),
                               zero_velocity, gt_velocity)
        velo_loss = model_util.huber_loss(
            pred_velocity - gt_velocity, delta=1.)
        velo_loss = tf.reduce_sum(velo_loss, axis=-1) * velo_mask
        velo_loss = tf.identity(tf.reduce_sum(
            velo_loss) / tf.maximum(1., tf.reduce_sum(velo_mask)), 'velocity_loss_%d' % index)
        tf.summary.scalar('velocity_loss_%d' % index, velo_loss)
        # tf.add_to_collection(tf.GraphKeys.LOSSES, velo_loss)
        return velo_loss

    def corner_loss(self, index, label_dict, pred_dict):
        pmask = label_dict[maps_dict.GT_PMASK][index]
        nmask = label_dict[maps_dict.GT_NMASK][index]
        gt_corners = label_dict[maps_dict.CORNER_LOSS_GT_BOXES_CORNERS][index]

        pred_corners = pred_dict[maps_dict.CORNER_LOSS_PRED_BOXES_CORNERS][index]

        norm_param = tf.maximum(1., tf.reduce_sum(pmask))

        corner_loss = model_util.huber_loss(
            (pred_corners - gt_corners), delta=1.)
        corner_loss = tf.reduce_sum(corner_loss, axis=[-2, -1]) * pmask
        corner_loss = tf.identity(tf.reduce_sum(
            corner_loss) / norm_param, 'corner_loss%d' % index)
        tf.summary.scalar('corner_loss%d' % index, corner_loss)
        # tf.add_to_collection(tf.GraphKeys.LOSSES, corner_loss)
        return corner_loss

    def offset_loss_res(self, index, label_dict, pred_dict):
        pmask = label_dict[maps_dict.GT_PMASK][index]
        nmask = label_dict[maps_dict.GT_NMASK][index]
        gt_offset = label_dict[maps_dict.GT_OFFSET][index]

        pred_offset = pred_dict[maps_dict.PRED_OFFSET][index]

        norm_param = tf.maximum(1., tf.reduce_sum(pmask))

        offset_loss = model_util.huber_loss(
            (pred_offset - gt_offset), delta=1.)
        offset_loss = tf.reduce_sum(offset_loss, axis=-1) * pmask
        offset_loss = tf.identity(tf.reduce_sum(
            offset_loss) / norm_param, 'offset_loss%d' % index)
        tf.summary.scalar('offset_loss%d' % index, offset_loss)
        # tf.add_to_collection(tf.GraphKeys.LOSSES, offset_loss)
        return offset_loss

    def offset_loss_bin(self, index, label_dict, pred_dict):
        pmask = label_dict[maps_dict.GT_PMASK][index]
        nmask = label_dict[maps_dict.GT_NMASK][index]

        # bs, points_num, cls_num, 8
        # xbin/xres/zbin/zres/yres/size_res
        gt_offset = label_dict[maps_dict.GT_OFFSET][index]
        xbin, xres, zbin, zres = tf.unstack(gt_offset[:, :, :, :4], axis=-1)
        gt_other_offset = gt_offset[:, :, :, 4:]

        pred_offset = pred_dict[maps_dict.PRED_OFFSET][index]
        pred_xbin = tf.slice(pred_offset, [
                             0, 0, 0, self.reg_bin_cls_num * 0], [-1, -1, -1, self.reg_bin_cls_num])
        pred_xres = tf.slice(pred_offset, [
                             0, 0, 0, self.reg_bin_cls_num * 1], [-1, -1, -1, self.reg_bin_cls_num])
        pred_zbin = tf.slice(pred_offset, [
                             0, 0, 0, self.reg_bin_cls_num * 2], [-1, -1, -1, self.reg_bin_cls_num])
        pred_zres = tf.slice(pred_offset, [
                             0, 0, 0, self.reg_bin_cls_num * 3], [-1, -1, -1, self.reg_bin_cls_num])
        pred_other_offset = tf.slice(
            pred_offset, [0, 0, 0, self.reg_bin_cls_num * 4], [-1, -1, -1, -1])

        norm_param = tf.maximum(1., tf.reduce_sum(pmask))

        self.bin_res_loss(pmask, norm_param, xbin, xres, pred_xbin,
                          pred_xres, self.reg_bin_cls_num, 'x_loss%d' % index)
        self.bin_res_loss(pmask, norm_param, zbin, zres, pred_zbin,
                          pred_zres, self.reg_bin_cls_num, 'z_loss%d' % index)

        other_offset_loss = model_util.huber_loss(
            (pred_other_offset - gt_other_offset), delta=1.)
        other_offset_loss = tf.reduce_sum(other_offset_loss, axis=-1) * pmask
        other_offset_loss = tf.identity(tf.reduce_sum(
            other_offset_loss) / norm_param, 'other_offset_loss%d' % index)
        tf.summary.scalar('other_offset_loss%d' % index, other_offset_loss)
        # tf.add_to_collection(tf.GraphKeys.LOSSES, other_offset_loss)
        return other_offset_loss

    def angle_loss(self, index, label_dict, pred_dict):
        # [bs, points_num, cls_num]
        gt_angle_cls = label_dict[maps_dict.GT_ANGLE_CLS][index]
        gt_angle_res = label_dict[maps_dict.GT_ANGLE_RES][index]
        pmask = label_dict[maps_dict.GT_PMASK][index]
        nmask = label_dict[maps_dict.GT_NMASK][index]

        # [bs, points_num, cls_num, cfg.MODEL.ANGLE_CLS_NUM]
        pred_angle_cls = pred_dict[maps_dict.PRED_ANGLE_CLS][index]
        pred_angle_res = pred_dict[maps_dict.PRED_ANGLE_RES][index]

        norm_param = tf.maximum(1., tf.reduce_sum(pmask))

        # self.bin_res_loss(pmask, norm_param, gt_angle_cls, gt_angle_res, pred_angle_cls, pred_angle_res,
        #                   cfg.MODEL.ANGLE_CLS_NUM, 'angle_loss%d' % index)
        bin_loss, res_loss = self.bin_res_loss(pmask, norm_param, gt_angle_cls, gt_angle_res,
                                               pred_angle_cls, pred_angle_res, cfg.MODEL.ANGLE_CLS_NUM, 'angle_loss%d' % index)
        return bin_loss, res_loss

    def bin_res_loss(self, pmask, norm_param, gt_bin, gt_res, pred_bin, pred_res, bin_class_num, scope):
        gt_bin = tf.cast(gt_bin, tf.int32)

        bin_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=pred_bin, labels=gt_bin) * pmask
        bin_loss = tf.identity(tf.reduce_sum(
            bin_loss) / norm_param, 'bin_%s' % scope)
        tf.summary.scalar('bin_%s' % scope, bin_loss)
        # tf.add_to_collection(tf.GraphKeys.LOSSES, bin_loss)

        gt_bin_onehot = tf.cast(tf.one_hot(
            gt_bin, depth=bin_class_num, on_value=1, off_value=0, axis=-1), tf.float32)
        pred_res = tf.reduce_sum(pred_res * gt_bin_onehot, axis=-1)
        res_loss = model_util.huber_loss((pred_res - gt_res) * pmask, delta=1.)
        res_loss = tf.identity(tf.reduce_sum(
            res_loss) / norm_param, 'res_%s' % scope)
        tf.summary.scalar('res_%s' % scope, res_loss)
        # tf.add_to_collection(tf.GraphKeys.LOSSES, res_loss)
        return bin_loss, res_loss
