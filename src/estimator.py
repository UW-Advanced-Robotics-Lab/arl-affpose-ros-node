#! /usr/bin/env python

import argparse
import os
import sys
sys.path.append('..')
import copy
import random
import numpy as np
import transformations
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

import tf
import tf2_ros
import tf2_geometry_msgs
import geometry_msgs.msg

import std_msgs.msg
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pcl2

##################################
### GPU
##################################
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cpu")

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

from src import kalman_filter as kf

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
        ### DenseFusion
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

        ##################################
        # TF transforms
        ##################################

        # Transforms
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)
        self.transform_broadcaster = tf2_ros.TransformBroadcaster()

        # WAM links / joints
        self.base_link_frame = 'wam/base_link'
        self.forearm_link_frame = 'wam/forearm_link'
        self.ee_link_frame = 'wam/wrist_palm_link'  # 'wam/bhand/bhand_grasp_link'
        self.zed_link_frame = 'zed_camera_center'  # 'zed_camera_center' or 'camera_frame'
        self.camera_link_frame = 'camera_frame'
        self.object_frame = 'object_frame'
        self.object_part_frame = 'object_part_frame'

        self.pub_pose  = rospy.Publisher('~aff_densefusion_pose', PoseStamped,queue_size=1)
        self.pub_object_models = rospy.Publisher('aff_densefusion_object_models', PointCloud2, queue_size=1)
        self.pub_object_part_models = rospy.Publisher('aff_densefusion_object_part_models', PointCloud2, queue_size=1)

        ##################################
        # Kalman filter
        ##################################

        rate = 5 # todo
        self.kalman_filter = kf.KalmanFilter(
            num_states=6,
            num_measurements=6,
            process_noise=1e-5,
            measurement_noise=1e-4,
            dt=1 / rate,
        )

        #########################
        #########################

        rospy.loginfo("Successfully loaded DenseFusion!\n")

    ##################################
    ##################################

    def get_6dof_pose(self, rgb, depth, pred_mask, pred_colour_mask):
        cld_img_pred = np.array(pred_colour_mask.copy())
        obj_ids = np.unique(pred_mask)[1:]
        for obj_id in obj_ids:
            if obj_id in self.class_IDs:
                rospy.loginfo("*** DenseFusion detected: {} ***".format(self.obj_classes[int(obj_id) - 1]))

                ##################################
                # BBOX
                ##################################
                x1, y1, x2, y2 = get_obj_bbox(pred_mask, obj_id, self.height, self.width)

                ##################################
                # MASK
                ##################################

                mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
                mask_label = ma.getmaskarray(ma.masked_equal(pred_mask, obj_id))
                mask = mask_label * mask_depth

                try:

                    ##################################
                    # Select Region of Interest
                    ##################################

                    choose = mask[y1:y2, x1:x2].flatten().nonzero()[0]

                    # print("Zero Division: ", self.num_points, len(choose))
                    if len(choose) == 0: # TODO !!!
                        rospy.loginfo("------ Could not detect Object Part ID: {} ------".format(obj_id))
                        return None
                    elif len(choose) >= self.num_points:
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

                    ###############################
                    ###############################

                    pred_q = my_r
                    pred_r = quaternion_matrix(pred_q)[0:3, 0:3]
                    pred_t = my_t

                    pred_rvec, _ = cv2.Rodrigues(pred_r)
                    pred_rvec = pred_rvec * 180 / np.pi
                    pred_rvec = np.squeeze(np.array(pred_rvec)).reshape(-1)

                    helper_utils.print_object_pose(t=pred_t, R=pred_r)

                    ###############################
                    # plotting
                    ###############################

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

                    ######################
                    # RVIZ
                    ######################

                    # tf
                    object_in_camera_frame = geometry_msgs.msg.TransformStamped()
                    object_in_camera_frame.header.frame_id = self.camera_link_frame
                    object_in_camera_frame.child_frame_id = self.object_frame
                    object_in_camera_frame.header.stamp = rospy.Time.now()
                    object_in_camera_frame.transform.translation.x = pred_t[0]
                    object_in_camera_frame.transform.translation.y = pred_t[1]
                    object_in_camera_frame.transform.translation.z = pred_t[2]
                    object_in_camera_frame.transform.rotation.w = pred_q[0]
                    object_in_camera_frame.transform.rotation.x = pred_q[1]
                    object_in_camera_frame.transform.rotation.y = pred_q[2]
                    object_in_camera_frame.transform.rotation.z = pred_q[3]
                    self.transform_broadcaster.sendTransform(object_in_camera_frame)

                    # pose
                    pose_msg = PoseStamped()
                    pose_msg.header.frame_id = self.camera_link_frame
                    pose_msg.pose.position.x = pred_t[0]
                    pose_msg.pose.position.y = pred_t[1]
                    pose_msg.pose.position.z = pred_t[2]
                    pose_msg.pose.orientation.w = pred_q[0]
                    pose_msg.pose.orientation.x = pred_q[1]
                    pose_msg.pose.orientation.y = pred_q[2]
                    pose_msg.pose.orientation.z = pred_q[3]
                    self.pub_pose.publish(pose_msg)

                    # pointcloud
                    header = std_msgs.msg.Header()
                    header.stamp = rospy.Time.now()
                    header.frame_id = self.camera_link_frame
                    _model_points = np.dot(self.cld[obj_id], pred_r.T) + pred_t
                    model_points = pcl2.create_cloud_xyz32(header, _model_points)
                    self.pub_object_models.publish(model_points)

                    ###############################
                    ###############################

                except ZeroDivisionError:  # ZeroDivisionError
                    rospy.loginfo("------ Could not detect Object Part ID: {} ------".format(obj_id))
                    pass

        return cld_img_pred

    ##################################
    ##################################

    def get_6dof_pose_kf(self, rgb, depth, pred_mask, pred_colour_mask):
        cld_img_pred = np.array(pred_colour_mask.copy())
        obj_ids = np.unique(pred_mask)[1:]
        for obj_id in obj_ids:
            if obj_id in self.class_IDs:
                rospy.loginfo("*** DenseFusion detected: {} ***".format(self.obj_classes[int(obj_id) - 1]))

                ##################################
                # BBOX
                ##################################
                x1, y1, x2, y2 = get_obj_bbox(pred_mask, obj_id, self.height, self.width)

                ##################################
                # MASK
                ##################################

                mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
                mask_label = ma.getmaskarray(ma.masked_equal(pred_mask, obj_id))
                mask = mask_label * mask_depth

                try:

                    ##################################
                    # Select Region of Interest
                    ##################################

                    choose = mask[y1:y2, x1:x2].flatten().nonzero()[0]

                    # print("Zero Division: ", self.num_points, len(choose))
                    if len(choose) == 0:  # TODO !!!
                        rospy.loginfo("------ Could not detect Object Part ID: {} ------".format(obj_id))
                        return None
                    elif len(choose) >= self.num_points:
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
                        T = Variable(torch.from_numpy(my_t.astype(np.float32))).cuda().view(1, 3).repeat(
                            self.num_points, 1).contiguous().view(1, self.num_points, 3)
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

                    ###############################
                    ###############################

                    pred_q = my_r
                    pred_r = quaternion_matrix(pred_q)[0:3, 0:3]
                    pred_t = my_t

                    pred_rvec, _ = cv2.Rodrigues(pred_r)
                    pred_rvec = pred_rvec * 180 / np.pi
                    pred_rvec = np.squeeze(np.array(pred_rvec)).reshape(-1)

                    ##################################
                    # Transforms
                    ##################################
                    try:

                        # pose
                        object_in_camera_frame_msg = PoseStamped()
                        # object_in_camera_frame_msg.header.frame_id = self.camera_link_frame
                        object_in_camera_frame_msg.pose.position.x = pred_t[0]
                        object_in_camera_frame_msg.pose.position.y = pred_t[1]
                        object_in_camera_frame_msg.pose.position.z = pred_t[2]
                        object_in_camera_frame_msg.pose.orientation.w = pred_q[0]
                        object_in_camera_frame_msg.pose.orientation.x = pred_q[1]
                        object_in_camera_frame_msg.pose.orientation.y = pred_q[2]
                        object_in_camera_frame_msg.pose.orientation.z = pred_q[3]

                        ''' object_T_world = object_T_zed * zed_T_world '''
                        # zed_T_world
                        camera_to_world = self.tf_buffer.lookup_transform(self.base_link_frame, self.camera_link_frame, rospy.Time(0))
                        # camera_to_world = self.tf_buffer.lookup_transform(self.camera_link_frame, self.base_link_frame, rospy.Time(0))
                        # object_T_world
                        object_to_world = tf2_geometry_msgs.do_transform_pose(object_in_camera_frame_msg, camera_to_world)

                        pred_world_t = np.array([object_to_world.pose.position.x,
                                                 object_to_world.pose.position.y,
                                                 object_to_world.pose.position.z])

                        pred_world_q = np.array([object_to_world.pose.orientation.w,
                                                 object_to_world.pose.orientation.x,
                                                 object_to_world.pose.orientation.y,
                                                 object_to_world.pose.orientation.z])

                        pred_world_r = quaternion_matrix(pred_world_q)[0:3, 0:3]

                        pred_object_pose_quaternion = helper_utils.convert_R_and_T_matrix_to_object_pose_quaternion(R=pred_world_r, T=pred_world_t)
                        helper_utils.print_object_pose(R=pred_world_r, t=pred_world_t)

                    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                        rospy.logwarn("Can't find transform from {} to {}".format(self.base_link_frame, self.camera_link_frame))
                        return None

                    ##################################
                    # Transforms
                    ##################################
                    pred_object_pose_vector = helper_utils.convert_object_pose_quaternion_to_object_pose_vector(pred_object_pose_quaternion)

                    if not self.kalman_filter._is_init:
                        self.kalman_filter.initialize(pred_object_pose_vector)
                    else:
                        self.kalman_filter.prediction()
                        kf_object_pose_vector = self.kalman_filter.correction(pred_object_pose_vector)
                        kf_world_r, kf_world_t = helper_utils.convert_object_pose_vector_to_R_and_t(kf_object_pose_vector)
                        helper_utils.print_object_pose(R=kf_world_r, t=kf_world_t, source='kf')

                    ###############################
                    # plotting
                    ###############################

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

                    ######################
                    # RVIZ
                    ######################

                    pred_world_t = np.array([0.75, 0, 0.25])
                    # pred_world_q = np.array([1 / np.sqrt(2), 1 / np.sqrt(2), 0, 0])
                    # pred_world_r = quaternion_matrix(pred_world_q)[0:3, 0:3]

                    # tf
                    object_in_world_frame = geometry_msgs.msg.TransformStamped()
                    object_in_world_frame.header.frame_id = self.base_link_frame
                    object_in_world_frame.child_frame_id = self.object_frame
                    object_in_world_frame.header.stamp = rospy.Time.now()
                    object_in_world_frame.transform.translation.x = pred_world_t[0]
                    object_in_world_frame.transform.translation.y = pred_world_t[1]
                    object_in_world_frame.transform.translation.z = pred_world_t[2]
                    object_in_world_frame.transform.rotation.w = pred_world_q[0]
                    object_in_world_frame.transform.rotation.x = pred_world_q[1]
                    object_in_world_frame.transform.rotation.y = pred_world_q[2]
                    object_in_world_frame.transform.rotation.z = pred_world_q[3]
                    self.transform_broadcaster.sendTransform(object_in_world_frame)

                    # pose
                    pose_msg = PoseStamped()
                    pose_msg.header.frame_id = self.base_link_frame
                    pose_msg.pose.position.x = pred_world_t[0]
                    pose_msg.pose.position.y = pred_world_t[1]
                    pose_msg.pose.position.z = pred_world_t[2]
                    pose_msg.pose.orientation.w = pred_world_q[0]
                    pose_msg.pose.orientation.x = pred_world_q[1]
                    pose_msg.pose.orientation.y = pred_world_q[2]
                    pose_msg.pose.orientation.z = pred_world_q[3]
                    self.pub_pose.publish(pose_msg)

                    # pointcloud
                    header = std_msgs.msg.Header()
                    header.stamp = rospy.Time.now()
                    header.frame_id = self.base_link_frame
                    _model_points = np.dot(self.cld[obj_id], pred_world_r.T) + pred_world_t
                    model_points = pcl2.create_cloud_xyz32(header, _model_points)
                    self.pub_object_models.publish(model_points)

                    ###############################
                    ###############################

                except ZeroDivisionError:  # ZeroDivisionError
                    rospy.loginfo("------ Could not detect Object Part ID: {} ------".format(obj_id))
                    pass

        return cld_img_pred

    ##################################
    ##################################

    def get_6dof_pose_gt(self, rgb, depth, pred_mask, pred_colour_mask, gt_meta, path_to_save, num_image):
        cld_img_pred = np.array(pred_colour_mask.copy())
        obj_ids = np.unique(pred_mask)[1:]
        gt_object_poses = []
        pred_object_poses = []
        kf_object_poses = []
        for obj_id in obj_ids:
            if obj_id in self.class_IDs:
                # rospy.loginfo('')
                rospy.loginfo("*** DenseFusion detected: {} ***".format(self.obj_classes[int(obj_id) - 1]))

                ##################################
                # BBOX
                ##################################
                x1, y1, x2, y2 = get_obj_bbox(pred_mask, obj_id, self.height, self.width)

                ##################################
                # MASK
                ##################################

                mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
                mask_label = ma.getmaskarray(ma.masked_equal(pred_mask, obj_id))
                mask = mask_label * mask_depth

                try:

                    ##################################
                    # Select Region of Interest
                    ##################################

                    choose = mask[y1:y2, x1:x2].flatten().nonzero()[0]

                    # print("Zero Division: ", self.num_points, len(choose))
                    if len(choose) == 0:  # TODO !!!
                        rospy.loginfo("------ Could not detect Object Part ID: {} ------".format(obj_id))
                        return None
                    elif len(choose) >= self.num_points:
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
                        T = Variable(torch.from_numpy(my_t.astype(np.float32))).cuda().view(1, 3).repeat(
                            self.num_points, 1).contiguous().view(1, self.num_points, 3)
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

                    ###############################
                    ###############################

                    pred_q = my_r
                    pred_r = quaternion_matrix(pred_q)[0:3, 0:3]
                    pred_t = my_t

                    pred_rvec, _ = cv2.Rodrigues(pred_r)
                    pred_rvec = pred_rvec * 180 / np.pi
                    pred_rvec = np.squeeze(np.array(pred_rvec)).reshape(-1)

                    ##################################
                    # GT
                    ##################################

                    obj_meta_idx = str(1000 + obj_id)[1:]
                    gt_r = np.array(gt_meta['obj_rotation_' + np.str(obj_meta_idx)]).reshape(3, 3)
                    gt_t = np.array(gt_meta['obj_translation_' + np.str(obj_meta_idx)]).reshape(-1)

                    gt_q = helper_utils.convert_R_and_T_matrix_to_object_pose_quaternion(R=gt_r, T=gt_t)[0:4]

                    ##################################
                    # Transforms
                    ##################################
                    try:

                        ##################################
                        # gt
                        ##################################
                        object_in_camera_frame_msg = PoseStamped()
                        # object_in_camera_frame_msg.header.frame_id = self.camera_link_frame
                        object_in_camera_frame_msg.pose.position.x = gt_t[0]
                        object_in_camera_frame_msg.pose.position.y = gt_t[1]
                        object_in_camera_frame_msg.pose.position.z = gt_t[2]
                        object_in_camera_frame_msg.pose.orientation.w = gt_q[0]
                        object_in_camera_frame_msg.pose.orientation.x = gt_q[1]
                        object_in_camera_frame_msg.pose.orientation.y = gt_q[2]
                        object_in_camera_frame_msg.pose.orientation.z = gt_q[3]

                        ''' object_T_world = object_T_zed * zed_T_world '''
                        # zed_T_world
                        camera_to_world = self.tf_buffer.lookup_transform(self.base_link_frame, self.camera_link_frame, rospy.Time(0))
                        # object_T_world
                        object_to_world = tf2_geometry_msgs.do_transform_pose(object_in_camera_frame_msg, camera_to_world)

                        gt_world_t = np.array([object_to_world.pose.position.x,
                                                 object_to_world.pose.position.y,
                                                 object_to_world.pose.position.z])

                        gt_world_q = np.array([object_to_world.pose.orientation.w,
                                               object_to_world.pose.orientation.x,
                                               object_to_world.pose.orientation.y,
                                               object_to_world.pose.orientation.z])

                        gt_world_r = quaternion_matrix(gt_world_q)[0:3, 0:3]

                        gt_object_pose_quaternion = helper_utils.convert_R_and_T_matrix_to_object_pose_quaternion(R=gt_world_r, T=gt_world_t)
                        gt_object_poses.append(gt_object_pose_quaternion)

                        helper_utils.print_object_pose(R=gt_world_r, t=gt_world_t, source='gt')

                        ##################################
                        # pred
                        ##################################
                        object_in_camera_frame_msg = PoseStamped()
                        # object_in_camera_frame_msg.header.frame_id = self.camera_link_frame
                        object_in_camera_frame_msg.pose.position.x = pred_t[0]
                        object_in_camera_frame_msg.pose.position.y = pred_t[1]
                        object_in_camera_frame_msg.pose.position.z = pred_t[2]
                        object_in_camera_frame_msg.pose.orientation.w = pred_q[0]
                        object_in_camera_frame_msg.pose.orientation.x = pred_q[1]
                        object_in_camera_frame_msg.pose.orientation.y = pred_q[2]
                        object_in_camera_frame_msg.pose.orientation.z = pred_q[3]

                        ''' object_T_world = object_T_zed * zed_T_world '''
                        # zed_T_world
                        camera_to_world = self.tf_buffer.lookup_transform(self.base_link_frame, self.camera_link_frame, rospy.Time(0))
                        # object_T_world
                        object_to_world = tf2_geometry_msgs.do_transform_pose(object_in_camera_frame_msg, camera_to_world)

                        pred_world_t = np.array([object_to_world.pose.position.x,
                                                 object_to_world.pose.position.y,
                                                 object_to_world.pose.position.z])

                        pred_world_q = np.array([object_to_world.pose.orientation.w,
                                                 object_to_world.pose.orientation.x,
                                                 object_to_world.pose.orientation.y,
                                                 object_to_world.pose.orientation.z])

                        pred_world_r = quaternion_matrix(pred_world_q)[0:3, 0:3]

                        pred_object_pose_quaternion = helper_utils.convert_R_and_T_matrix_to_object_pose_quaternion(R=pred_world_r, T=pred_world_t)
                        pred_object_poses.append(pred_object_pose_quaternion)

                        helper_utils.print_object_pose(R=pred_world_r, t=pred_world_t)

                    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                        rospy.logwarn("Can't find transform from {} to {}".format(self.base_link_frame, self.camera_link_frame))
                        return None

                    ##################################
                    # Transforms
                    ##################################
                    pred_object_pose_vector = helper_utils.convert_object_pose_quaternion_to_object_pose_vector(pred_object_pose_quaternion)

                    if not self.kalman_filter._is_init:
                        self.kalman_filter.initialize(pred_object_pose_vector)
                        kf_object_poses.append(pred_object_pose_quaternion)
                    else:
                        self.kalman_filter.prediction()
                        kf_object_pose_vector = self.kalman_filter.correction(pred_object_pose_vector)

                        # errors
                        kf_world_r, kf_world_t = helper_utils.convert_object_pose_vector_to_R_and_t(kf_object_pose_vector)
                        kf_object_pose_quaternion = helper_utils.convert_R_and_T_matrix_to_object_pose_quaternion(R=kf_world_r, T=kf_world_t)
                        kf_object_poses.append(kf_object_pose_quaternion)

                        helper_utils.print_object_pose(R=kf_world_r, t=kf_world_t, source='kf')

                        helper_utils.quantify_errors(gt_r=gt_world_r, gt_t=gt_world_t,
                                                     pred_r=kf_world_r, pred_t=kf_world_t,
                                                     pose_method='Kalman Filter')

                    ###############################
                    # plotting
                    ###############################

                    cld_img_pred = helper_utils.draw_object_pose(np.array(pred_colour_mask.copy()),
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

                    ######################
                    # RVIZ
                    ######################

                    # tf
                    object_in_world_frame = geometry_msgs.msg.TransformStamped()
                    object_in_world_frame.header.frame_id = self.base_link_frame
                    object_in_world_frame.child_frame_id = self.object_frame
                    object_in_world_frame.header.stamp = rospy.Time.now()
                    object_in_world_frame.transform.translation.x = pred_world_t[0]
                    object_in_world_frame.transform.translation.y = pred_world_t[1]
                    object_in_world_frame.transform.translation.z = pred_world_t[2]
                    object_in_world_frame.transform.rotation.w = pred_world_q[0]
                    object_in_world_frame.transform.rotation.x = pred_world_q[1]
                    object_in_world_frame.transform.rotation.y = pred_world_q[2]
                    object_in_world_frame.transform.rotation.z = pred_world_q[3]
                    self.transform_broadcaster.sendTransform(object_in_world_frame)

                    # pose
                    pose_msg = PoseStamped()
                    pose_msg.header.frame_id = self.base_link_frame
                    pose_msg.pose.position.x = pred_world_t[0]
                    pose_msg.pose.position.y = pred_world_t[1]
                    pose_msg.pose.position.z = pred_world_t[2]
                    pose_msg.pose.orientation.w = pred_world_q[0]
                    pose_msg.pose.orientation.x = pred_world_q[1]
                    pose_msg.pose.orientation.y = pred_world_q[2]
                    pose_msg.pose.orientation.z = pred_world_q[3]
                    self.pub_pose.publish(pose_msg)

                    # pointcloud
                    header = std_msgs.msg.Header()
                    header.stamp = rospy.Time.now()
                    header.frame_id = self.base_link_frame
                    _model_points = np.dot(self.cld[obj_id], pred_world_r.T) + pred_world_t
                    model_points = pcl2.create_cloud_xyz32(header, _model_points)
                    self.pub_object_models.publish(model_points)

                    ###############################
                    ###############################

                except ZeroDivisionError:  # ZeroDivisionError
                    rospy.loginfo("------ Could not detect Object Part ID: {} ------".format(obj_id))
                    pass

        ######################
        # TODO: MATLAB
        ######################

        scio.savemat('{0}/{1}.mat'.format(path_to_save, '%04d' % num_image),
                     {"class_ids": obj_ids,
                      'gt': gt_object_poses,
                      'pred': pred_object_poses,
                      'kf': kf_object_poses,
                      })

        return cld_img_pred

        ##################################
        ##################################

    def transform_obj_pose(self, gt_mask, gt_colour_mask, gt_r, gt_t, gt_r_world, gt_t_world):
        cld_img_pred = np.array(gt_colour_mask.copy())
        obj_ids = np.unique(gt_mask)[1:]
        for obj_id in obj_ids:
            if obj_id in self.class_IDs:

                ##################################
                # Transforms
                ##################################
                try:

                    gt_q_world = helper_utils.convert_R_and_T_matrix_to_object_pose_quaternion(R=gt_r_world, T=gt_t_world)[0:4]

                    # pose
                    object_in_camera_frame_msg = PoseStamped()
                    # object_in_camera_frame_msg.header.frame_id = self.camera_link_frame
                    object_in_camera_frame_msg.pose.position.x = gt_t_world[0]
                    object_in_camera_frame_msg.pose.position.y = gt_t_world[1]
                    object_in_camera_frame_msg.pose.position.z = gt_t_world[2]
                    object_in_camera_frame_msg.pose.orientation.w = gt_q_world[0]
                    object_in_camera_frame_msg.pose.orientation.x = gt_q_world[1]
                    object_in_camera_frame_msg.pose.orientation.y = gt_q_world[2]
                    object_in_camera_frame_msg.pose.orientation.z = gt_q_world[3]

                    ''' object_T_world = object_T_zed * zed_T_world '''
                    # zed_T_world
                    camera_to_world = self.tf_buffer.lookup_transform(self.base_link_frame, self.camera_link_frame,
                                                                      rospy.Time(0))
                    # object_T_world
                    object_to_world = tf2_geometry_msgs.do_transform_pose(object_in_camera_frame_msg,
                                                                          camera_to_world)

                    gt_world_t = np.array([object_to_world.pose.position.x,
                                           object_to_world.pose.position.y,
                                           object_to_world.pose.position.z])

                    gt_world_q = np.array([object_to_world.pose.orientation.w,
                                           object_to_world.pose.orientation.x,
                                           object_to_world.pose.orientation.y,
                                           object_to_world.pose.orientation.z])

                    gt_world_r = quaternion_matrix(gt_world_q)[0:3, 0:3]
                    helper_utils.print_object_pose(R=gt_world_r, t=gt_world_t)

                except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                    rospy.logwarn(
                        "Can't find transform from {} to {}".format(self.base_link_frame, self.camera_link_frame))
                    return None

                ###############################
                # plotting
                ###############################

                gt_rvec, _ = cv2.Rodrigues(gt_r)
                gt_rvec = gt_rvec * 180 / np.pi
                gt_rvec = np.squeeze(np.array(gt_rvec)).reshape(-1)

                cld_img_pred = helper_utils.draw_object_pose(cld_img_pred,
                                                             self.cld[obj_id] * 1e3,
                                                             gt_r,
                                                             gt_t * 1e3,
                                                             self.cam_mat,
                                                             self.cam_dist,
                                                             obj_color=(255, 255, 0))

                cld_img_pred = helper_utils.put_position_orientation_value_to_frame(cld_img_pred,
                                                                                    gt_rvec,
                                                                                    gt_t.copy())

                ######################
                # RVIZ - OBJECT
                ######################

                # tf
                object_in_world_frame = geometry_msgs.msg.TransformStamped()
                object_in_world_frame.header.frame_id = self.base_link_frame
                object_in_world_frame.child_frame_id = self.object_frame
                object_in_world_frame.header.stamp = rospy.Time.now()
                object_in_world_frame.transform.translation.x = gt_world_t[0]
                object_in_world_frame.transform.translation.y = gt_world_t[1]
                object_in_world_frame.transform.translation.z = gt_world_t[2]
                object_in_world_frame.transform.rotation.w = gt_world_q[0]
                object_in_world_frame.transform.rotation.x = gt_world_q[1]
                object_in_world_frame.transform.rotation.y = gt_world_q[2]
                object_in_world_frame.transform.rotation.z = gt_world_q[3]
                self.transform_broadcaster.sendTransform(object_in_world_frame)

                # pose
                pose_msg = PoseStamped()
                pose_msg.header.frame_id = self.base_link_frame
                pose_msg.pose.position.x = gt_world_t[0]
                pose_msg.pose.position.y = gt_world_t[1]
                pose_msg.pose.position.z = gt_world_t[2]
                pose_msg.pose.orientation.w = gt_world_q[0]
                pose_msg.pose.orientation.x = gt_world_q[1]
                pose_msg.pose.orientation.y = gt_world_q[2]
                pose_msg.pose.orientation.z = gt_world_q[3]
                self.pub_pose.publish(pose_msg)

                # pointcloud
                header = std_msgs.msg.Header()
                header.stamp = rospy.Time.now()
                header.frame_id = self.base_link_frame
                _model_points = np.dot(self.cld[obj_id], gt_world_r.T) + gt_world_t
                model_points = pcl2.create_cloud_xyz32(header, _model_points)
                self.pub_object_models.publish(model_points)

                #######################
                # Transforms: OBJ to OBJ PART
                #######################

                obj_part_offset_in_world = np.array([-0.000242, -0.107078, 0.000337])
                obj_part_offset_in_object_frame = obj_part_offset_in_world.copy()
                # x axis: is +ive y axis in world frame
                obj_part_offset_in_object_frame[0] = obj_part_offset_in_world[1]
                # y axis: is -ive z axis in world frame
                obj_part_offset_in_object_frame[1] = -1 * obj_part_offset_in_world[2]
                # z axis: is -ive x axis in world frame
                obj_part_offset_in_object_frame[2] = -1 * obj_part_offset_in_world[0]

                self.obj_part_frame = geometry_msgs.msg.TransformStamped()
                self.obj_part_frame.header.stamp = rospy.Time.now()
                self.obj_part_frame.header.frame_id = self.object_frame
                self.obj_part_frame.child_frame_id = self.object_part_frame
                self.obj_part_frame.transform.translation.x = obj_part_offset_in_object_frame[0]
                self.obj_part_frame.transform.translation.y = obj_part_offset_in_object_frame[1]
                self.obj_part_frame.transform.translation.z = obj_part_offset_in_object_frame[2]
                self.obj_part_frame.transform.rotation.w = 1
                self.obj_part_frame.transform.rotation.x = 0
                self.obj_part_frame.transform.rotation.y = 0
                self.obj_part_frame.transform.rotation.z = 0
                self.transform_broadcaster.sendTransform(self.obj_part_frame)

                try:

                    obj_part_to_cam = self.tf_buffer.lookup_transform(self.camera_link_frame, self.object_part_frame, rospy.Time(0))

                    obj_part_to_cam_t = np.array([obj_part_to_cam.transform.translation.x,
                                                  obj_part_to_cam.transform.translation.y,
                                                  obj_part_to_cam.transform.translation.z])

                    obj_part_to_cam_q = np.array([obj_part_to_cam.transform.rotation.w,
                                                  obj_part_to_cam.transform.rotation.x,
                                                  obj_part_to_cam.transform.rotation.y,
                                                  obj_part_to_cam.transform.rotation.z])

                    obj_part_to_cam_r = quaternion_matrix(obj_part_to_cam_q)[0:3, 0:3]

                    pose_msg = PoseStamped()
                    pose_msg.header.frame_id = self.object_frame
                    pose_msg.pose.position.x = obj_part_offset_in_object_frame[0]
                    pose_msg.pose.position.y = obj_part_offset_in_object_frame[1]
                    pose_msg.pose.position.z = obj_part_offset_in_object_frame[2]
                    pose_msg.pose.orientation.w = 1
                    pose_msg.pose.orientation.x = 0
                    pose_msg.pose.orientation.y = 0
                    pose_msg.pose.orientation.z = 0

                    object_to_camera = self.tf_buffer.lookup_transform(self.camera_link_frame, self.object_frame, rospy.Time(0))

                    _obj_part_to_cam = tf2_geometry_msgs.do_transform_pose(pose_msg, object_to_camera)

                    _obj_part_to_cam_t = np.array([_obj_part_to_cam.pose.position.x,
                                           _obj_part_to_cam.pose.position.y,
                                           _obj_part_to_cam.pose.position.z])

                    _obj_part_to_cam_q = np.array([_obj_part_to_cam.pose.orientation.w,
                                           _obj_part_to_cam.pose.orientation.x,
                                           _obj_part_to_cam.pose.orientation.y,
                                           _obj_part_to_cam.pose.orientation.z])

                    _obj_part_to_cam_r = quaternion_matrix(_obj_part_to_cam_q)[0:3, 0:3]

                    print("obj_part_to_cam_t: ", obj_part_to_cam_t)
                    print("_obj_part_to_cam_t: ", _obj_part_to_cam_t)

                except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                    rospy.logwarn("Can't find transform from {} to {}".format(self.base_link_frame, self.object_part_frame))
                    return None

        return cld_img_pred
