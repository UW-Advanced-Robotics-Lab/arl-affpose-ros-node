#!/usr/bin/env python
from __future__ import division

import os
import sys
sys.path.append('..')
import copy

import numpy as np
import cv2

import skimage.draw
from skimage import data, color, io, img_as_float

import rospy

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data

from torchvision.transforms import functional as F

##################################
##################################

from affnet.MaskRCNN import ResNetMaskRCNN

import affnet.cfg as config
from utils import helper_utils

from utils.dataset import vicon_dataset_utils

##################################
##################################

class Detector(object):
    def __init__(self, model_path, num_classes):

        rospy.loginfo("Loading AffNet .. ")

        self.transform = self.get_transform()
        self.model = ResNetMaskRCNN(pretrained=False, num_classes=num_classes+1) # +1 for background
        self.model.to(config.DEVICE)

        # print("\nrestoring pre-trained MaskRCNN weights: {} .. ".format(config.RESTORE_TRAINED_WEIGHTS))
        checkpoint = torch.load(model_path, map_location=config.DEVICE)
        self.model.load_state_dict(checkpoint["model"])

        self.model.eval()
        rospy.loginfo("Successfully loaded AffNet!\n")

    def get_transform(self):
        transforms = []
        transforms.append(ToTensor())
        return Compose(transforms)

    def detect_bbox_and_mask(self, img):
        # img = copy.deepcopy(images)
        images, _ = self.transform(image=img, target={})
        images = list(image.to(config.DEVICE) for image in images)

        with torch.no_grad():
            outputs = self.model(images)
            outputs = [{k: v.to(config.CPU_DEVICE) for k, v in t.items()} for t in outputs]

            #######################
            ### todo: formatting output
            #######################
            outputs = outputs.pop()

            scores = np.array(outputs['scores'], dtype=np.float32).flatten()

            if len(scores) == 0:
                return None, None
            else:
                idx = np.argwhere(scores.copy() > config.CONFIDENCE_THRESHOLD)
                scores = scores[idx]

            labels = np.array(outputs['labels'], dtype=np.int32).flatten()
            boxes = np.array(outputs['boxes'], dtype=np.int32).reshape(-1, 4)
            labels = labels[idx]
            boxes = boxes[idx]

            binary_masks = np.squeeze(np.array(outputs['masks'] > config.CONFIDENCE_THRESHOLD, dtype=np.uint8))

            aff_labels = labels.copy()
            if 'aff_labels' in outputs.keys():
                aff_labels = np.array(outputs['aff_labels'], dtype=np.int32)

            #######################
            ### bbox
            #######################
            bbox_img = helper_utils.draw_bbox_on_img(image=img,
                                                     labels=labels,
                                                     boxes=boxes,
                                                     scores=scores)

            #######################
            ### masks
            #######################
            mask = helper_utils.get_segmentation_masks(image=bbox_img,
                                                       labels=labels,
                                                       binary_masks=binary_masks,
                                                       scores=scores)

            pred_colour_mask = vicon_dataset_utils.colorize_obj_mask(mask)
            pred_colour_mask = cv2.addWeighted(bbox_img, 0.35, pred_colour_mask, 0.65, 0)

            #####################
            # MASKED RGB IMG
            #####################

            # cv2.imshow('bbox', cv2.cvtColor(bbox_img, cv2.COLOR_BGR2RGB))
            # cv2.imshow('pred_colour_mask', cv2.cvtColor(pred_colour_mask, cv2.COLOR_BGR2RGB))
            # cv2.imshow('mask_color_img', cv2.cvtColor(mask_color_img, cv2.COLOR_BGR2RGB))
            # cv2.waitKey(0)

            #####################
            #####################

            return mask, pred_colour_mask

##################################
##################################

class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

