#! /usr/bin/env python
from __future__ import division
'''
This ros node subscribes to two camera topics: '/camera/color/image_raw' and 
'/camera/aligned_depth_to_color/image_raw' in a synchronized way. It then runs 
semantic segmentation and pose estimation with trained models using DenseFusion
(https://github.com/j96w/DenseFusion). The whole code structure is adapted from: 
(http://wiki.ros.org/rospy_tutorials/Tutorials/WritingImagePublisherSubscriber)
'''

import os
import sys
sys.path.append('..')
import glob
import time

import numpy as np
from PIL import Image as PILImage
import scipy.io as scio

from scipy.spatial.transform import Rotation as R

import cv2
from cv_bridge import CvBridge, CvBridgeError

import rospy

import message_filters
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CameraInfo, Image as ImageSensor_msg
from std_msgs.msg import String
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose
from visualization_msgs.msg import Marker, MarkerArray

from sensor_msgs.msg import PointCloud2
import std_msgs.msg
import sensor_msgs.point_cloud2 as pcl2

##################################
##################################

from estimator import DenseFusionEstimator
from segmentation import Detector

from utils import helper_utils

from utils.bbox.extract_bboxs_from_label import get_obj_bbox

from utils.dataset import vicon_dataset_utils

##################################
##################################

class AffordanceDetector(Detector):

    def __init__(self, model_path, num_classes):
        Detector.__init__(self, model_path, num_classes)

##################################
##################################

class PoseEstimator(DenseFusionEstimator):

    def __init__(self):

        ##################################
        # ZED Camera
        ##################################

        self.__rgb_image = rospy.get_param('~rgb_image', None)
        self.__rgb_encoding = rospy.get_param('~rgb_encoding', None)
        self.__depth_image = rospy.get_param('~depth_image', None)
        self.__depth_encoding = rospy.get_param('~depth_encoding', None)

        self.__cam_scale = rospy.get_param('~cam_scale', None)

        self.__cam_width = rospy.get_param('~cam_width', None)
        self.__cam_height = rospy.get_param('~cam_height', None)
        self.__resize = (int(self.__cam_width), int(self.__cam_height))
        self.__crop_width = rospy.get_param('~crop_width', None)
        self.__crop_height = rospy.get_param('~crop_height', None)
        self.__crop_size = (int(self.__crop_width), int(self.__crop_height))

        self.__x_scale = self.__crop_width / self.__cam_width
        self.__y_scale = self.__crop_height / self.__cam_height

        # self.__cam_cx = rospy.get_param('~cam_cx', None)
        # self.__cam_cy = rospy.get_param('~cam_cy', None)
        self.__cam_cx = rospy.get_param('~cam_cx', None) * self.__x_scale
        self.__cam_cy = rospy.get_param('~cam_cy', None) * self.__y_scale
        self.__cam_fx = rospy.get_param('~cam_fx', None)
        self.__cam_fy = rospy.get_param('~cam_fy', None)

        # Image Processing
        self.__blur_threshold = rospy.get_param('~blur_threshold', None)

        ##################################
        # Densefusion
        ##################################

        self.__classes = rospy.get_param('~classes', None)
        self.__class_ids = rospy.get_param('~class_ids', None)
        assert os.path.isfile(self.__classes), 'File not found! {}'.format(self.__classes)
        assert os.path.isfile(self.__class_ids), 'File not found! {}'.format(self.__class_ids)

        self.__pose_model = rospy.get_param('~pose_model', None)
        self.__refine_model = rospy.get_param('~refine_model', None)
        assert os.path.isfile(self.__pose_model), 'File not found! {}'.format(self.__pose_model)
        assert os.path.isfile(self.__refine_model), 'File not found! {}'.format(self.__refine_model)

        self.__num_points = rospy.get_param('~num_points', None)
        self.__num_points_mesh = rospy.get_param('~num_points_mesh', None)
        self.__iteration = rospy.get_param('~iteration', None)
        self.__bs = rospy.get_param('~bs', None)
        self.__num_obj = rospy.get_param('~num_obj', None)


        DenseFusionEstimator.__init__(self,
                                      classes_file_=self.__classes,
                                      class_ids_file_=self.__class_ids,
                                      model=self.__pose_model,
                                      refine_model=self.__refine_model,
                                      num_points=self.__num_points, num_points_mesh=self.__num_points_mesh,
                                      iteration=self.__iteration, bs=self.__bs, num_obj=self.__num_obj,
                                      cam_width=self.__crop_width, cam_height=self.__crop_height,
                                      cam_scale=self.__cam_scale,
                                      cam_fx=self.__cam_fx, cam_fy=self.__cam_fy,
                                      cam_cx=self.__cam_cx, cam_cy=self.__cam_cy)

        ##################################
        # AffNet
        ##################################
        self.__affnet_model = rospy.get_param('~affnet_model', None)
        assert os.path.isfile(self.__affnet_model), 'File not found! {}'.format(self.__affnet_model)

        self.AffordanceDetector = AffordanceDetector(self.__affnet_model, self.__num_obj)

        ##################################
        # Publisher
        ##################################

        self.pub_rgb   = rospy.Publisher('~aff_densefusion_rgb', Image, queue_size=1)
        self.pub_depth = rospy.Publisher('~aff_densefusion_depth', Image, queue_size=1)
        self.pub_mask  = rospy.Publisher('~aff_densefusion_mask', Image, queue_size=1)
        self.pub_pred = rospy.Publisher('~aff_densefusion_pred', Image, queue_size=1)
        self.pub_pose  = rospy.Publisher('~aff_densefusion_pose', PoseStamped,queue_size=1)
        # self.pub_densefusion_model_points = rospy.Publisher('densefusion_model_points', PointCloud2, queue_size=1)

        ##################################
        # Testing
        ##################################

        self.num_image = 0
        self.max_num_images = rospy.get_param('~max_num_images', None)
        self.test_image_paths = rospy.get_param('~test_image_paths', None)
        self.image_path = rospy.get_param('~saved_image_path', None)

        self.rgb_path = self.image_path + 'rgb/'
        if not os.path.exists(self.rgb_path):
            os.makedirs(self.rgb_path)
        files = glob.glob(self.rgb_path + '*')
        for file in files:
            os.remove(file)
        self.depth_path = self.image_path + 'depth/'
        if not os.path.exists(self.depth_path):
            os.makedirs(self.depth_path)
        files = glob.glob(self.depth_path + '*')
        for file in files:
            os.remove(file)
        self.mask_path = self.image_path + 'mask/'
        if not os.path.exists(self.mask_path):
            os.makedirs(self.mask_path)
        files = glob.glob(self.mask_path + '*')
        for file in files:
            os.remove(file)
        self.pose_path = self.image_path + 'pose/'
        if not os.path.exists(self.pose_path):
            os.makedirs(self.pose_path)
        files = glob.glob(self.pose_path + '*')
        for file in files:
            os.remove(file)

        ##################################
        # Callback
        ##################################

        self.bridge = CvBridge()
        self.rgb_sub = message_filters.Subscriber(self.__rgb_image, Image)
        self.depth_sub = message_filters.Subscriber(self.__depth_image, Image)
        ts = message_filters.TimeSynchronizer([self.rgb_sub, self.depth_sub], 1)
        ts.registerCallback(self.camera_callback)
        rospy.loginfo('Subscribed to rgb and depth topic in a sychronized way!\n')

    ######################
    ######################
    def camera_callback(self, rgb_msg, depth_msg):

        if self.num_image < self.max_num_images:
            self.num_image += 1
        else:
            self.num_image = 1
        # rospy.loginfo('image:{}/{} ..'.format(self.num_image, self.max_num_images))

        #########################
        ### Load Images with ROS
        #########################

        rgb_cv = self.bridge.imgmsg_to_cv2(rgb_msg, self.__rgb_encoding)
        rgb_cv = self.bridge.cv2_to_imgmsg(rgb_cv, self.__rgb_encoding)
        bgr = np.frombuffer(rgb_cv.data, dtype=np.uint8).reshape(rgb_cv.height, rgb_cv.width, -1)
        rgb = np.array(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

        depth_cv = self.bridge.imgmsg_to_cv2(depth_msg, self.__depth_encoding)  # "16UC1" or "32FC1"
        depth_cv = self.bridge.cv2_to_imgmsg(depth_cv, self.__depth_encoding)
        depth_16bit = np.frombuffer(depth_cv.data, dtype=np.uint16).reshape(rgb_cv.height, rgb_cv.width)

        # helper_utils.print_depth_info(depth_16bit)
        # depth_8bit = helper_utils.convert_16_bit_depth_to_8_bit(depth_16bit)
        # helper_utils.print_depth_info(depth_8bit)

        #########################
        ### Test images
        #########################

        # num_str = np.str(10000000000 + self.num_image)[1:]
        # rgb_addr = self.test_image_paths + num_str + '_rgb.png'
        # rgb = PILImage.open(rgb_addr).convert('RGB')
        # rgb = np.array(rgb, dtype=np.uint8)
        #
        # depth_addr = self.test_image_paths + num_str + '_depth.png'
        # depth_16bit = cv2.imread(depth_addr, -1)

        ##################################
        # RESIZE & CROP
        ##################################

        # RESIZE & CROP
        rgb = cv2.resize(rgb, self.__resize, interpolation=cv2.INTER_CUBIC)
        rgb = helper_utils.crop(pil_img=rgb, crop_size=self.__crop_size, is_img=True)

        depth_16bit = cv2.resize(depth_16bit, self.__resize, interpolation=cv2.INTER_CUBIC)
        depth_16bit = helper_utils.crop(pil_img=depth_16bit, crop_size=self.__crop_size)

        ##################################
        # blur value
        ##################################
        if helper_utils.is_blur(rgb, blur_threshold=self.__blur_threshold):
            return

        ##################################
        # RVIZ
        ##################################

        cv2_rgb = self.bridge.cv2_to_imgmsg(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), self.__rgb_encoding)
        self.pub_rgb.publish(cv2_rgb)

        cv2_depth_16bit = self.bridge.cv2_to_imgmsg(depth_16bit, self.__depth_encoding)
        self.pub_depth.publish(cv2_depth_16bit)

        # SAVE
        rgb_addr = self.rgb_path + np.str(self.num_image) + '.png'
        cv2.imwrite(rgb_addr, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        depth_addr = self.depth_path + np.str(self.num_image) + '.png'
        cv2.imwrite(depth_addr, np.array(depth_16bit).astype(np.uint16))

        ######################
        # Segmentation
        ######################

        rospy.loginfo("\n")
        rospy.loginfo('Segmentation start ..')
        t_start = time.time()
        pred_mask, mask_color_img = self.AffordanceDetector.detect_bbox_and_mask(rgb)
        t_segmentation = time.time() - t_start
        rospy.loginfo('Segmentation Prediction time: {:.2f}s'.format(t_segmentation))

        cv2_mask_color_img = self.bridge.cv2_to_imgmsg(cv2.cvtColor(mask_color_img, cv2.COLOR_BGR2RGB), self.__rgb_encoding)
        self.pub_mask.publish(cv2_mask_color_img)

        pred_mask_addr = self.mask_path + np.str(self.num_image) + '_pred.png'
        cv2.imwrite(pred_mask_addr, pred_mask)

        ######################
        # DenseFusion
        ######################

        rospy.loginfo("")
        rospy.loginfo('DenseFusion start ..')
        t_start = time.time()
        pred_R, pred_T, pred_img = DenseFusionEstimator.get_refined_pose(self, rgb, depth_16bit, pred_mask, mask_color_img)
        t_densefusion = time.time() - t_start
        rospy.loginfo('DenseFusion: pred T: {}'.format(pred_T))
        rospy.loginfo('DenseFusion: pred R: {}'.format(pred_R))
        rospy.loginfo('DenseFusion Prediction time: {:.2f}s'.format(t_densefusion))

        cv2_pred_img = self.bridge.cv2_to_imgmsg(cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB), self.__rgb_encoding)
        self.pub_pred.publish(cv2_pred_img)

        ######################
        # RVIZ
        ######################

        # pose
        pose_msg = PoseStamped()
        pose_msg.header = rgb_msg.header
        pose_msg.pose.position.x = pred_T[0]
        pose_msg.pose.position.y = pred_T[1]
        pose_msg.pose.position.z = pred_T[2]
        pose_msg.pose.orientation.x = pred_R[0]
        pose_msg.pose.orientation.y = pred_R[1]
        pose_msg.pose.orientation.z = pred_R[2]
        pose_msg.pose.orientation.w = pred_R[3]
        self.pub_pose.publish(pose_msg)

        # # pointcloud
        # header = std_msgs.msg.Header()
        # header.stamp = rospy.Time.now()
        # header.frame_id = rgb_msg.header
        # densefusion_point_cloud = pcl2.create_cloud_xyz32(rgb_msg.header, model_points)
        # self.pub_densefusion_model_points.publish(densefusion_point_cloud)


def main(args):

    rospy.init_node('learning_pose_estimation', anonymous=True)

    _rate = rospy.get_param('~rate', None)
    _rate = -1. if _rate <= 0. else _rate
    rate = rospy.Rate(_rate)  # 5 Hz

    print('')
    rospy.loginfo('Running node at {} Hz ..\n'.format(_rate))

    PoseEstimator()
    while not rospy.is_shutdown():
        rospy.spin()
        if rate > 0.:
            rate.sleep()
    # except KeyboardInterrupt:
    #     print ('Shutting down ROS pose estimation module')
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)