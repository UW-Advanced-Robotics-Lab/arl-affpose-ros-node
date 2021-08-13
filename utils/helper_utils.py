import sys
sys.path.append('..')
import os
from os import listdir
from os.path import splitext
from glob import glob

import copy

import numpy as np
import transformations

import cv2
from PIL import Image

import rospy

import torch
import torch.nn.functional as F

##################################
##################################

from utils.dataset import vicon_dataset_utils

##################################
# FORMAT UTILS
##################################

def print_depth_info(depth):
    depth = np.array(depth)
    rospy.loginfo("Depth of type:{} has min:{} & max:{}".format(depth.dtype, np.min(depth), np.max(depth)))

def convert_16_bit_depth_to_8_bit(depth):
    depth = np.array(depth, np.uint16)
    # depth = depth / np.max(depth) * (2 ** 8 - 1)
    return np.array(depth, np.uint8)

def format_label(label):
    return np.array(label, dtype=np.int32)

def format_bbox(bbox):
    return np.array(bbox, dtype=np.int32).flatten()

def crop(pil_img, crop_size, is_img=False):
    _dtype = np.array(pil_img).dtype
    pil_img = Image.fromarray(pil_img)
    crop_w, crop_h = crop_size
    img_width, img_height = pil_img.size
    left, right = (img_width - crop_w) / 2, (img_width + crop_w) / 2
    top, bottom = (img_height - crop_h) / 2, (img_height + crop_h) / 2
    left, top = round(max(0, left)), round(max(0, top))
    right, bottom = round(min(img_width - 0, right)), round(min(img_height - 0, bottom))
    # pil_img = pil_img.crop((left, top, right, bottom)).resize((crop_w, crop_h))
    pil_img = pil_img.crop((left, top, right, bottom))
    ###
    if is_img:
        img_channels = np.array(pil_img).shape[-1]
        img_channels = 3 if img_channels == 4 else img_channels
        resize_img = np.zeros((crop_w, crop_h, img_channels))
        resize_img[0:(int(bottom) - int(top)), 0:(int(right) - int(left)), :img_channels] = np.array(pil_img)[..., :img_channels]
    else:
        resize_img = np.zeros((crop_w, crop_h))
        resize_img[0:(int(bottom) - int(top)), 0:(int(right) - int(left))] = np.array(pil_img)

    return np.array(resize_img, dtype=_dtype)

######################
# IMG UTILS
######################

def is_blur(rgb, blur_threshold):
    # cv2.imshow("rgb", cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
    # cv2.waitKey(0)

    blur_val = cv2.Laplacian(rgb, cv2.CV_64F).var()
    if blur_val < blur_threshold:
        rospy.loginfo('Blur Value: {} for RGB frame is too high ..'.format(blur_val))
        return True
    else:
        return False

######################
# AffNet UTILS
######################

def get_segmentation_masks(image, labels, binary_masks, scores=None, is_gt=False):

    height, width = image.shape[:2]
    # print(f'height:{height}, width:{width}')

    instance_masks = np.zeros((height, width), dtype=np.uint8)
    instance_mask_one = np.ones((height, width), dtype=np.uint8)

    if len(binary_masks.shape) == 2:
        binary_masks = binary_masks[np.newaxis, :, :]

    for idx, label in enumerate(labels):
        label = labels[idx]
        binary_mask = np.array(binary_masks[idx, :, :], dtype=np.uint8)
        instance_mask = instance_mask_one * label
        instance_masks = np.where(binary_mask, instance_mask, instance_masks).astype(np.uint8)

    # print_class_labels(instance_masks)
    return instance_masks

def draw_bbox_on_img(image, labels, boxes, scores=None, confidence_threshold=0.35):
    bbox_img = image.copy()

    for idx, score in enumerate(scores):
        if score > confidence_threshold:
            bbox = format_bbox(boxes[idx])
            bbox_img = cv2.rectangle(bbox_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 255, 1)

            label = labels[idx]
            cv2.putText(bbox_img,
                        # elevator_utils.object_id_to_name(label),
                        vicon_dataset_utils.map_obj_id_to_name(label),
                        (bbox[0], bbox[1] - 5),
                        cv2.FONT_ITALIC,
                        0.6,
                        (255, 255, 255))

    return bbox_img

#####################
# Pose Estimation Utils
######################

def get_pose_in_world_frame(object_r, object_t, camera_r, camera_t):
    ''' object_T_world = object_T_zed * zed_T_world '''

    #######################################
    # OBJECT in WORLD FRAME
    #######################################

    # zed_T_world
    world_T_camera = np.eye(4)
    world_T_camera[0:3, 0:3] = camera_r
    world_T_camera[0:3, -1] = camera_t

    #######################################
    # pred
    #######################################

    # object_T_zed
    camera_T_object = np.eye(4)
    camera_T_object[0:3, 0:3] = object_r
    camera_T_object[0:3, -1] = object_t

    ''' object_T_world = object_T_zed * zed_T_world '''
    world_T_object = np.dot(world_T_camera, camera_T_object)
    R = world_T_object[0:3, 0:3].reshape(3, 3)
    T = world_T_object[0:3, -1].reshape(-1)

    return R, T

def convert_R_and_T_matrix_to_object_pose_quaternion(R, T):
    '''
    tvec: x, y, z in [m]
    rvec: roll, pitch, yaw in [deg]
    '''

    T = np.array(T).reshape(-1)
    R = np.array(R).reshape(3, 3)

    SE3 = np.eye(4)
    SE3[0:3, 0:3] = R

    quat = transformations.quaternion_from_matrix(SE3).reshape(-1)

    object_pose_quaternion = np.hstack([quat, T.reshape(-1)])
    return object_pose_quaternion

def convert_object_pose_quaternion_to_object_pose_vector(object_pose_quaternion):
    '''
    tvec: x, y, z in [m]
    rvec: roll, pitch, yaw in [deg]
    '''

    # Rotation Matrix
    R = transformations.quaternion_matrix(object_pose_quaternion[:4].reshape(-1))[0:3, 0:3]
    rvec, _ = cv2.Rodrigues(R)
    # rotation vector: convert to degree
    rvec = rvec * 180 / np.pi

    # tvec
    T = object_pose_quaternion[4:]
    T = T.reshape(-1)

    object_pose_vector = np.hstack([rvec.reshape(-1), T.reshape(-1)]).reshape(-1)
    return object_pose_vector

def convert_object_pose_vector_to_R_and_t(object_pose_vector):
    '''
    tvec: x, y, z in [m]
    rvec: roll, pitch, yaw in [deg]
    '''

    # Rotation Matrix
    rvec = object_pose_vector[:3] * np.pi / 180
    R, _ = cv2.Rodrigues(rvec)

    # tvec
    T = object_pose_vector[3:]
    T = T.reshape(-1)

    return R, T

def quantify_errors(gt_r, gt_t, pred_r, pred_t, pose_method='PnP'):

    #######################################
    # format
    #######################################

    gt_t = np.array(gt_t).reshape(-1)
    pred_t = np.array(pred_t).reshape(-1)
    gt_r = np.array(gt_r).reshape(3, 3)
    pred_r = np.array(pred_r).reshape(3, 3)

    #######################################
    #######################################

    # translation
    pred_T_error = np.linalg.norm(pred_t - gt_t)
    pred_T_error_x = np.linalg.norm(pred_t[0] - gt_t[0])
    pred_T_error_y = np.linalg.norm(pred_t[1] - gt_t[1])
    pred_T_error_z = np.linalg.norm(pred_t[2] - gt_t[2])

    # rot
    error_cos = 0.5 * (np.trace(np.dot(pred_r, np.linalg.inv(gt_r))) - 1.0)
    error_cos = min(1.0, max(-1.0, error_cos))
    error = np.arccos(error_cos)
    pred_R_error = 180.0 * error / np.pi

    rospy.loginfo("*** Quantifying Errors for: {} ***".format(pose_method))
    rospy.loginfo("translation [cm] : {:.2f}".format(pred_T_error*100))
    rospy.loginfo("rotation    [deg]: {:.2f}".format(pred_R_error))

######################
# DenseFusion UTILS
######################

def sort_imgpts(_imgpts):
    imgpts = np.squeeze(_imgpts.copy())
    imgpts = imgpts[np.lexsort(np.transpose(imgpts)[::-1])]
    return np.int32([imgpts])

def draw_object_pose(rgb, pointcloud, R, T, cam_mat, cam_dist, obj_color=(255, 0, 0)):
    pose_img = rgb.copy()
    # project point cloud
    imgpts, jac = cv2.projectPoints(pointcloud, R, T, cam_mat, cam_dist)
    # pose_img = cv2.polylines(pose_img, sort_imgpts(imgpts), True, obj_color)

    # draw pose
    rotV, _ = cv2.Rodrigues(R)
    points = np.float32([[100, 0, 0], [0, 100, 0], [0, 0, 100], [0, 0, 0]]).reshape(-1, 3)
    axisPoints, _ = cv2.projectPoints(points, rotV, T, cam_mat, cam_dist)
    pose_img = cv2.line(pose_img, tuple(axisPoints[3].ravel()),tuple(axisPoints[0].ravel()), (255, 0, 0), 3)
    pose_img = cv2.line(pose_img, tuple(axisPoints[3].ravel()),tuple(axisPoints[1].ravel()), (0, 255, 0), 3)
    pose_img = cv2.line(pose_img, tuple(axisPoints[3].ravel()),tuple(axisPoints[2].ravel()), (0, 0, 255), 3)

    return pose_img

def put_position_orientation_value_to_frame(rgb, R, T):
    pose_img = rgb.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    # convert translation to [cm]
    T *= 100

    cv2.putText(pose_img, 'orientation [degree]', (10, 60), font, 0.7, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(pose_img, 'x:' + str(round(R[0], 2)), (250, 60), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(pose_img, 'y:' + str(round(R[1], 2)), (350, 60), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(pose_img, 'z:' + str(round(R[2], 2)), (450, 60), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(pose_img, 'position [cm]', (10, 30), font, 0.7, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(pose_img, 'x:' + str(round(T[0], 2)), (250, 30), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(pose_img, 'y:' + str(round(T[1], 2)), (350, 30), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(pose_img, 'z:' + str(round(T[2], 2)), (450, 30), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    return pose_img

def print_object_pose(t, R, source='pred'):

    # convert translation to [cm]
    t = t.copy() * 100

    rvec, _ = cv2.Rodrigues(R)
    rvec = rvec * 180 / np.pi
    rvec = np.squeeze(np.array(rvec)).reshape(-1)

    rospy.loginfo('{}: position    [cm]:  x:{:.2f}, y:{:.2f}, z:{:.2f}'.format(source, t[0], t[1], t[2]))
    rospy.loginfo('{}: orientation [deg]: x:{:.2f}, y:{:.2f}, z:{:.2f}'.format(source, rvec[0], rvec[1], rvec[2]))