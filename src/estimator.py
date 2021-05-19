#! /usr/bin/env python

import argparse
import os
import sys
sys.path.append('..')
import copy
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import scipy.io as scio
import scipy.misc
import numpy.ma as ma
import math

np.seterr(divide='ignore')

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.functional as F
from torch.autograd import Variable

import rospy

##################################
### GPU
##################################
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# device = torch.device("cpu")

print("\n********* Torch GPU ************")
print("torch.__version__:", torch.__version__)
print("current_device :", torch.cuda.current_device())
print("cuda.device:", torch.cuda.device(0))
print("cuda.device_count:", torch.cuda.device_count())
print("get_device_name:", torch.cuda.get_device_name(0))
print("cuda.is_available:", torch.cuda.is_available())
print("cuda.current_device:", torch.cuda.current_device())

##################################
# lib local to src for ros-2.7 env
##################################

from lib.network import PoseNet, PoseRefineNet
from lib.transformations import euler_matrix, quaternion_matrix, quaternion_from_matrix
from lib.knn.__init__ import KNearestNeighbor

knn = KNearestNeighbor(1)

##################################
##################################

from utils import helper_utils

from utils.dataset import vicon_dataset_utils

from utils.bbox.extract_bboxs_from_label import get_obj_bbox
from utils.pose.load_obj_ply_files import load_obj_ply_files

##################################
##################################

class DenseFusionEstimator():

    def __init__(self, model, refine_model,
                 num_points, num_points_mesh, iteration, bs, num_obj,
                 classes_file_, class_ids_file_,
                 cam_width, cam_height, cam_scale, cam_fx, cam_fy, cam_cx, cam_cy):

        #########################
        ### Load 3D Object Models
        #########################
        self.cld, self.obj_classes, self.class_IDs = load_obj_ply_files(classes_file_, class_ids_file_)

        #########################
        ### Camera Params
        #########################
        # TODO: need norm ?
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.width = cam_width
        self.height = cam_height

        self.xmap = np.array([[j for i in range(self.height)] for j in range(self.width)])
        self.ymap = np.array([[i for i in range(self.height)] for j in range(self.width)])

        self.cam_scale = cam_scale
        self.cam_fx = cam_fx
        self.cam_fy = cam_fy
        self.cam_cx = cam_cx
        self.cam_cy = cam_cy
        self.cam_mat = np.array([[self.cam_fx, 0, self.cam_cx], [0, self.cam_fy, self.cam_cy], [0, 0, 1]])
        self.cam_dist = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

        #########################
        ### Load Images with ROS
        #########################
        self.num_points = num_points
        self.num_points_mesh = num_points_mesh
        self.iteration = iteration
        self.bs = bs
        self.num_obj = num_obj

        self.estimator = PoseNet(num_points=self.num_points, num_obj=self.num_obj)
        self.estimator.cuda()
        self.estimator.load_state_dict(torch.load(model))
        self.estimator.eval()

        self.refiner = PoseRefineNet(num_points=self.num_points, num_obj=self.num_obj)
        self.refiner.cuda()
        self.refiner.load_state_dict(torch.load(refine_model))
        self.refiner.eval()

        rospy.loginfo("Successfully loaded DenseFusion!\n")

    ##################################
    ##################################

    def get_refined_pose(self, rgb, depth, pred_mask, mask_color_img):

        obj_ids = np.unique(pred_mask)[1:]
        for obj_id in obj_ids:
            if obj_id in self.class_IDs:
                try:
                    rospy.loginfo("*** DenseFusion detect Object Part ID: {} ***".format(obj_id))

                    ##################################
                    # MASK
                    ##################################

                    mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
                    mask_label = ma.getmaskarray(ma.masked_equal(pred_mask, obj_id))
                    mask = mask_label * mask_depth

                    ##################################
                    # BBOX
                    ##################################
                    # x1, y1, x2, y2 = bboxs[0], bboxs[1], bboxs[2], bboxs[3]
                    x1, y1, x2, y2 = get_obj_bbox(pred_mask, obj_id, self.height, self.width)

                    ##################################
                    # Select Region of Interest
                    ##################################

                    choose = mask[y1:y2, x1:x2].flatten().nonzero()[0]

                    # print("Zero Division: ", self.num_points, len(choose))
                    if len(choose) >= self.num_points:
                        c_mask = np.zeros(len(choose), dtype=int)
                        c_mask[:self.num_points] = 1
                        np.random.shuffle(c_mask)
                        choose = choose[c_mask.nonzero()]
                    else:
                        choose = np.pad(choose, (0, self.num_points - len(choose)), 'wrap')

                    depth_masked = depth[y1:y2, x1:x2].flatten()[choose][:, np.newaxis].astype(np.float32)
                    xmap_masked = self.xmap[y1:y2, x1:x2].flatten()[choose][:, np.newaxis].astype(np.float32)
                    ymap_masked = self.ymap[y1:y2, x1:x2].flatten()[choose][:, np.newaxis].astype(np.float32)
                    choose = np.array([choose])

                    ######################################
                    # create point cloud from depth image
                    ######################################

                    pt2 = depth_masked / self.cam_scale
                    pt0 = (ymap_masked - self.cam_cx) * pt2 / self.cam_fx
                    pt1 = (xmap_masked - self.cam_cy) * pt2 / self.cam_fy
                    cloud = np.concatenate((pt0, pt1, pt2), axis=1)

                    ######################################
                    ######################################

                    img_masked = np.array(rgb)[:, :, :3]
                    img_masked = np.transpose(img_masked, (2, 0, 1))
                    img_masked = img_masked[:, y1:y2, x1:x2]

                    cloud = torch.from_numpy(cloud.astype(np.float32))
                    choose = torch.LongTensor(choose.astype(np.int32))
                    # img_masked = self.norm(torch.from_numpy(img_masked.astype(np.float32)))
                    img_masked = self.norm(torch.from_numpy(img_masked.astype(np.float32)))
                    index = torch.LongTensor([obj_id - 1])

                    cloud = Variable(cloud).cuda()
                    choose = Variable(choose).cuda()
                    img_masked = Variable(img_masked).cuda()
                    index = Variable(index).cuda()

                    cloud = cloud.view(1, self.num_points, 3)
                    img_masked = img_masked.view(1, 3, img_masked.size()[1], img_masked.size()[2])

                    #######################################
                    #######################################

                    pred_r, pred_t, pred_c, emb = self.estimator(img_masked, cloud, choose, index)
                    pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, self.num_points, 1)

                    pred_c = pred_c.view(self.bs, self.num_points)
                    how_max, which_max = torch.max(pred_c, 1)
                    # print("how_max: {:.5f}".format(how_max.detach().cpu().numpy()[0]))

                    pred_t = pred_t.view(self.bs * self.num_points, 1, 3)
                    points = cloud.view(self.bs * self.num_points, 1, 3)

                    my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
                    my_t = (points + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
                    # print("my_pred w/o refinement: \n", my_pred)

                    for ite in range(0, self.iteration):
                        T = Variable(torch.from_numpy(my_t.astype(np.float32))).cuda().view(1, 3).repeat(self.num_points,1).contiguous().view(1, self.num_points, 3)
                        my_mat = quaternion_matrix(my_r)
                        R = Variable(torch.from_numpy(my_mat[:3, :3].astype(np.float32))).cuda().view(1, 3, 3)
                        my_mat[0:3, 3] = my_t

                        new_cloud = torch.bmm((cloud - T), R).contiguous()
                        pred_r, pred_t = self.refiner(new_cloud, emb, index)
                        pred_r = pred_r.view(1, 1, -1)
                        pred_r = pred_r / (torch.norm(pred_r, dim=2).view(1, 1, 1))
                        my_r_2 = pred_r.view(-1).cpu().data.numpy()
                        my_t_2 = pred_t.view(-1).cpu().data.numpy()
                        my_mat_2 = quaternion_matrix(my_r_2)

                        my_mat_2[0:3, 3] = my_t_2

                        my_mat_final = np.dot(my_mat, my_mat_2)
                        my_r_final = copy.deepcopy(my_mat_final)
                        my_r_final[0:3, 3] = 0
                        my_r_final = quaternion_from_matrix(my_r_final, True)
                        my_t_final = np.array([my_mat_final[0][3], my_mat_final[1][3], my_mat_final[2][3]])

                        my_pred = np.append(my_r_final, my_t_final)
                        my_r = my_r_final
                        my_t = my_t_final

                    # print("my_pred w/ {} refinement: \n{}".format(ite, my_pred))
                    pred_q = my_r
                    pred_r = quaternion_matrix(pred_q)[0:3, 0:3]
                    pred_t = my_t

                    pred_rvec, _ = cv2.Rodrigues(pred_r)
                    pred_rvec = pred_rvec * 180 / np.pi
                    pred_rvec = pred_rvec.reshape(-1)

                    ###############################
                    # plotting
                    ###############################

                    cld_img_pred = helper_utils.draw_object_pose(np.array(mask_color_img.copy()),
                                                                 self.cld[obj_id] * 1e3,
                                                                 pred_r,
                                                                 pred_t * 1e3,
                                                                 self.cam_mat,
                                                                 self.cam_dist,
                                                                 obj_color=(255, 255, 0))

                    cld_img_pred = helper_utils.put_position_orientation_value_to_frame(cld_img_pred,
                                                                                        pred_rvec,
                                                                                        pred_t.copy())

                    ###############################
                    ###############################

                    # cld_img_pred = cv2.cvtColor(cld_img_pred, cv2.COLOR_RGB2BGR)
                    # cv2.imshow('cld_img_pred', cv2.cvtColor(cld_img_pred, cv2.COLOR_BGR2RGB))
                    # cv2.waitKey(0)

                    ###############################
                    ###############################

                    return pred_q, pred_t, cld_img_pred

                except ZeroDivisionError:
                    rospy.loginfo("------ Could not detect Object Part ID: {} ------".format(obj_id))
                    return None, None, None, None

            else:
                rospy.loginfo("------ Could not detect Object Part ID: {} ------".format(obj_id))
                return None, None, None, None

    ##################################
    ##################################

    def get_refined_pose_gt(self, rgb, depth, pred_mask, mask_color_img, gt_mask, gt_r, gt_t):

        obj_ids = np.unique(pred_mask)[1:]
        for obj_id in obj_ids:
            if obj_id in self.class_IDs:
                try:
                    rospy.loginfo("*** DenseFusion detect Object Part ID: {} ***".format(obj_id))

                    ##################################
                    # MASK
                    ##################################

                    mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
                    mask_label = ma.getmaskarray(ma.masked_equal(pred_mask, obj_id))
                    mask = mask_label * mask_depth

                    ##################################
                    # BBOX
                    ##################################
                    # x1, y1, x2, y2 = bboxs[0], bboxs[1], bboxs[2], bboxs[3]
                    x1, y1, x2, y2 = get_obj_bbox(pred_mask, obj_id, self.height, self.width)

                    # bbox_img = np.array(rgb.copy())
                    # cv2.rectangle(bbox_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    # cv2.putText(bbox_img,
                    #             # elevator_utils.object_id_to_name(label),
                    #             vicon_dataset_utils.map_obj_id_to_name(obj_id),
                    #             (x1, y1 - 5),
                    #             cv2.FONT_ITALIC,
                    #             0.4,
                    #             (255, 255, 255))
                    #
                    # cv2.imshow('bbox', cv2.cvtColor(bbox_img, cv2.COLOR_BGR2RGB))
                    # cv2.waitKey(0)

                    ##################################
                    # Select Region of Interest
                    ##################################

                    choose = mask[y1:y2, x1:x2].flatten().nonzero()[0]

                    # print("Zero Division: ", self.num_points, len(choose))
                    if len(choose) >= self.num_points:
                        c_mask = np.zeros(len(choose), dtype=int)
                        c_mask[:self.num_points] = 1
                        np.random.shuffle(c_mask)
                        choose = choose[c_mask.nonzero()]
                    else:
                        choose = np.pad(choose, (0, self.num_points - len(choose)), 'wrap')

                    depth_masked = depth[y1:y2, x1:x2].flatten()[choose][:, np.newaxis].astype(np.float32)
                    xmap_masked = self.xmap[y1:y2, x1:x2].flatten()[choose][:, np.newaxis].astype(np.float32)
                    ymap_masked = self.ymap[y1:y2, x1:x2].flatten()[choose][:, np.newaxis].astype(np.float32)
                    choose = np.array([choose])

                    ######################################
                    # create point cloud from depth image
                    ######################################

                    pt2 = depth_masked / self.cam_scale
                    pt0 = (ymap_masked - self.cam_cx) * pt2 / self.cam_fx
                    pt1 = (xmap_masked - self.cam_cy) * pt2 / self.cam_fy
                    cloud = np.concatenate((pt0, pt1, pt2), axis=1)

                    ######################################
                    ######################################

                    img_masked = np.array(rgb)[:, :, :3]
                    img_masked = np.transpose(img_masked, (2, 0, 1))
                    img_masked = img_masked[:, y1:y2, x1:x2]

                    cloud = torch.from_numpy(cloud.astype(np.float32))
                    choose = torch.LongTensor(choose.astype(np.int32))
                    # img_masked = self.norm(torch.from_numpy(img_masked.astype(np.float32)))
                    img_masked = self.norm(torch.from_numpy(img_masked.astype(np.float32)))
                    index = torch.LongTensor([obj_id - 1])

                    cloud = Variable(cloud).cuda()
                    choose = Variable(choose).cuda()
                    img_masked = Variable(img_masked).cuda()
                    index = Variable(index).cuda()

                    cloud = cloud.view(1, self.num_points, 3)
                    img_masked = img_masked.view(1, 3, img_masked.size()[1], img_masked.size()[2])

                    #######################################
                    #######################################

                    pred_r, pred_t, pred_c, emb = self.estimator(img_masked, cloud, choose, index)
                    pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, self.num_points, 1)

                    pred_c = pred_c.view(self.bs, self.num_points)
                    how_max, which_max = torch.max(pred_c, 1)
                    # print("how_max: {:.5f}".format(how_max.detach().cpu().numpy()[0]))

                    pred_t = pred_t.view(self.bs * self.num_points, 1, 3)
                    points = cloud.view(self.bs * self.num_points, 1, 3)

                    my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
                    my_t = (points + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
                    # print("my_pred w/o refinement: \n", my_pred)

                    for ite in range(0, self.iteration):
                        T = Variable(torch.from_numpy(my_t.astype(np.float32))).cuda().view(1, 3).repeat(self.num_points,1).contiguous().view(1, self.num_points, 3)
                        my_mat = quaternion_matrix(my_r)
                        R = Variable(torch.from_numpy(my_mat[:3, :3].astype(np.float32))).cuda().view(1, 3, 3)
                        my_mat[0:3, 3] = my_t

                        new_cloud = torch.bmm((cloud - T), R).contiguous()
                        pred_r, pred_t = self.refiner(new_cloud, emb, index)
                        pred_r = pred_r.view(1, 1, -1)
                        pred_r = pred_r / (torch.norm(pred_r, dim=2).view(1, 1, 1))
                        my_r_2 = pred_r.view(-1).cpu().data.numpy()
                        my_t_2 = pred_t.view(-1).cpu().data.numpy()
                        my_mat_2 = quaternion_matrix(my_r_2)

                        my_mat_2[0:3, 3] = my_t_2

                        my_mat_final = np.dot(my_mat, my_mat_2)
                        my_r_final = copy.deepcopy(my_mat_final)
                        my_r_final[0:3, 3] = 0
                        my_r_final = quaternion_from_matrix(my_r_final, True)
                        my_t_final = np.array([my_mat_final[0][3], my_mat_final[1][3], my_mat_final[2][3]])

                        my_pred = np.append(my_r_final, my_t_final)
                        my_r = my_r_final
                        my_t = my_t_final

                    # print("my_pred w/ {} refinement: \n{}".format(ite, my_pred))
                    pred_r = quaternion_matrix(my_r)[0:3, 0:3]
                    pred_t = my_t

                    pred_rvec, _ = cv2.Rodrigues(pred_r)
                    pred_rvec = pred_rvec * 180 / np.pi
                    pred_rvec = pred_rvec.reshape(-1)

                    ###############################
                    # plotting
                    ###############################

                    cld_img_pred = helper_utils.draw_object_pose(np.array(mask_color_img.copy()),
                                                                 self.cld[obj_id] * 1e3,
                                                                 gt_r,
                                                                 gt_t * 1e3,
                                                                 self.cam_mat,
                                                                 self.cam_dist,
                                                                 obj_color=(0, 255, 255))


                    cld_img_pred = helper_utils.draw_object_pose(cld_img_pred,
                                                                 self.cld[obj_id] * 1e3,
                                                                 pred_r,
                                                                 pred_t * 1e3,
                                                                 self.cam_mat,
                                                                 self.cam_dist,
                                                                 obj_color=(255, 255, 0))

                    cld_img_pred = helper_utils.put_position_orientation_value_to_frame(cld_img_pred,
                                                                                        pred_rvec,
                                                                                        pred_t.copy())

                    ###############################
                    ###############################

                    # cld_img_pred = cv2.cvtColor(cld_img_pred, cv2.COLOR_RGB2BGR)
                    # cv2.imshow('cld_img_pred', cv2.cvtColor(cld_img_pred, cv2.COLOR_BGR2RGB))
                    # cv2.waitKey(0)

                    ###############################
                    ###############################

                    return pred_r, pred_t, cld_img_pred

                except ZeroDivisionError:
                    rospy.loginfo("------ Could not detect Object Part ID: {} ------".format(obj_id))
                    return None, None, None, None

            else:
                rospy.loginfo("------ Could not detect Object Part ID: {} ------".format(obj_id))
                return None, None, None, None