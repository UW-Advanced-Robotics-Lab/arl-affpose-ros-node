from collections import OrderedDict

import numpy as np
import cv2

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.model_zoo import load_url
from torchvision import models
from torchvision.ops import misc

###########################
###########################

from pathlib import Path
ROOT_DIR_PATH = Path(__file__).parents[1]

import affnet.cfg as config

###########################
###########################

from affnet.utils.transform_utils import Transformer
from affnet.utils.bbox_utils import AnchorGenerator

from affnet.FeatureExtractor import ResNetBackbone
from affnet.RPN import RPNHead, RegionProposalNetwork
from affnet.RoIAlign import RoIAlign
from affnet.RoIHeads import RoIHeads

from torchvision.ops import MultiScaleRoIAlign

###########################
###########################

class MaskRCNN(nn.Module):
    """
    Implements Mask R-CNN.

    The input image to the model is expected to be a tensor, shape [C, H, W], and should be in 0-1 range.

    The behavior of the model changes depending if it is in training or evaluation mode.

    During training, the model expects both the input tensor, as well as a target (dictionary),
    containing:
        - boxes (FloatTensor[N, 4]): the ground-truth boxes in [xmin, ymin, xmax, ymax] format, with values
          between 0-H and 0-W
        - labels (Int64Tensor[N]): the class label for each ground-truth box
        - masks (UInt8Tensor[N, H, W]): the segmentation binary masks for each instance

    The model returns a Dict[Tensor], containing the classification and regression losses 
    for both the RPN and the R-CNN, and the mask loss.

    During inference, the model requires only the input tensor, and returns the post-processed
    predictions as a Dict[Tensor]. The fields of the Dict are as
    follows:
        - boxes (FloatTensor[N, 4]): the predicted boxes in [xmin, ymin, xmax, ymax] format, 
          with values between 0-H and 0-W
        - labels (Int64Tensor[N]): the predicted labels
        - scores (FloatTensor[N]): the scores for each prediction
        - masks (FloatTensor[N, H, W]): the predicted masks for each instance, in 0-1 range. In order to
          obtain the final segmentation masks, the soft masks can be thresholded, generally
          with a value of 0.5 (mask >= 0.5)
        
    Arguments:
        backbone (nn.Module): the network used to compute the features for the affnet.
        num_classes (int): number of output classes of the model (including the background).
        
        rpn_fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training of the RPN.
        rpn_bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training of the RPN.
        rpn_num_samples (int): number of anchors that are sampled during training of the RPN
            for computing the loss
        rpn_positive_fraction (float): proportion of positive anchors during training of the RPN
        rpn_reg_weights (Tuple[float, float, float, float]): weights for the encoding/decoding of the
            bounding boxes
        rpn_pre_nms_top_n_train (int): number of proposals to keep before applying NMS during training
        rpn_pre_nms_top_n_test (int): number of proposals to keep before applying NMS during testing
        rpn_post_nms_top_n_train (int): number of proposals to keep after applying NMS during training
        rpn_post_nms_top_n_test (int): number of proposals to keep after applying NMS during testing
        rpn_nms_thresh (float): NMS threshold used for postprocessing the RPN proposals
        
        box_fg_iou_thresh (float): minimum IoU between the proposals and the GT box so that they can be
            considered as positive during training of the classification head
        box_bg_iou_thresh (float): maximum IoU between the proposals and the GT box so that they can be
            considered as negative during training of the classification head
        box_num_samples (int): number of proposals that are sampled during training of the
            classification head
        box_positive_fraction (float): proportion of positive proposals during training of the 
            classification head
        box_reg_weights (Tuple[float, float, float, float]): weights for the encoding/decoding of the
            bounding boxes
        box_score_thresh (float): during inference, only return proposals with a classification score
            greater than box_score_thresh
        box_nms_thresh (float): NMS threshold for the prediction head. Used during inference
        box_num_detections (int): maximum number of detections, for all classes.
        
    """
    
    def __init__(self, backbone, num_classes,
                 # Anchor Generator
                 anchor_sizes=config.ANCHOR_SIZES,
                 anchor_ratios=config.ANCHOR_RATIOS,
                 # RPN parameters
                 rpn_fg_iou_thresh=config.RPN_FG_IOU_THRESH,
                 rpn_bg_iou_thresh=config.RPN_BG_IOU_THRESH,
                 rpn_num_samples=config.RPN_NUM_SAMPLES,
                 rpn_positive_fraction=config.RPN_POSITIVE_FRACTION,
                 rpn_reg_weights=config.RPN_REG_WEIGHTS,
                 rpn_pre_nms_top_n_train=config.RPN_PRE_NMS_TOP_N_TRAIN,
                 rpn_pre_nms_top_n_test=config.RPN_PRE_NMS_TOP_N_TEST,
                 rpn_post_nms_top_n_train=config.RPN_POST_NMS_TOP_N_TRAIN,
                 rpn_post_nms_top_n_test=config.RPN_POST_NMS_TOP_N_TEST,
                 rpn_nms_thresh=config.RPN_NMS_THRESH,
                 # RoIHeads parameters
                 box_fg_iou_thresh=config.BOX_FG_IOU_THRESH,
                 box_bg_iou_thresh=config.BOX_BG_IOU_THRESH,
                 box_num_samples=config.BOX_NUM_SAMPLES,
                 box_positive_fraction=config.BOX_POSITIVE_FRACTION,
                 box_reg_weights=config.BOX_REG_WEIGHTS,
                 box_score_thresh=config.BOX_SCORE_THRESH,
                 box_nms_thresh=config.BOX_NMS_THRESH,
                 box_num_detections=config.BOX_NUM_DETECTIONS
                 ):
        super(MaskRCNN, self).__init__()

        ###############################
        ### Transformer
        ###############################

        self.transformer = Transformer(
            min_size=config.MIN_SIZE,
            max_size=config.MAX_SIZE,
            image_mean=config.IMAGE_MEAN,
            image_std=config.IMAGE_STD)

        ###############################
        ### Backbone
        ###############################

        self.backbone = backbone
        # out_channels = backbone.out_channels
        out_channels = 512 # ResNet50: 256 or Vgg16:512

        ###############################
        ### RPN
        ###############################

        num_anchors = len(anchor_sizes) * len(anchor_ratios)
        rpn_anchor_generator = AnchorGenerator(anchor_sizes, anchor_ratios)
        rpn_head = RPNHead(out_channels, num_anchors)
        
        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)
        self.rpn = RegionProposalNetwork(
             rpn_anchor_generator, rpn_head, 
             rpn_fg_iou_thresh, rpn_bg_iou_thresh,
             rpn_num_samples, rpn_positive_fraction,
             rpn_reg_weights,
             rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)

        ###############################
        ### RoIHeads
        ###############################

        box_roi_pool = RoIAlign(output_size=config.ROIALIGN_BOX_OUTPUT_SIZE,
                                sampling_ratio=config.ROIALIGN_SAMPLING_RATIO)
        
        resolution = box_roi_pool.output_size[0]
        in_channels = out_channels * resolution ** 2
        mid_channels = 1024
        box_predictor = FastRCNNPredictor(in_channels, mid_channels, num_classes)
        # box_predictor = FastRCNNPredictor(in_channels, mid_channels, config.NUM_OBJECT_CLASSES)

        self.head = RoIHeads(
             box_roi_pool, box_predictor,
             box_fg_iou_thresh, box_bg_iou_thresh,
             box_num_samples, box_positive_fraction,
             box_reg_weights,
             box_score_thresh, box_nms_thresh, box_num_detections)
        
        self.head.mask_roi_pool = RoIAlign(output_size=config.ROIALIGN_MASK_OUTPUT_SIZE,
                                           sampling_ratio=config.ROIALIGN_SAMPLING_RATIO)
        
        layers = (256, 256, 256, 256)
        dim_reduced = 256
        self.head.mask_predictor = MaskRCNNPredictor(out_channels, layers, dim_reduced, num_classes)
        # self.head.mask_predictor = MaskRCNNPredictor(out_channels, layers, dim_reduced, config.NUM_AFF_CLASSES)
        
    def forward(self, image, target=None):
        if isinstance(image, list):
            image = image.pop()
        if isinstance(target, list):
            target = target.pop()

        ori_image_shape = image.shape[-2:]

        image, target = self.transformer(image, target)
        image_shape = image.shape[-2:]
        feature = self.backbone(image)
        
        proposal, rpn_losses = self.rpn(feature, image_shape, target)
        result, roi_losses = self.head(feature, proposal, image_shape, target)
        
        if self.training:
            # return dict(**rpn_losses, **roi_losses)
            return dict(**rpn_losses), dict(**roi_losses)
        else:
            result = self.transformer.postprocess(result, image_shape, ori_image_shape)
            return list([result])

###############################
###############################
        
class FastRCNNPredictor(nn.Module):
    def __init__(self, in_channels, mid_channels, num_classes):
        super(FastRCNNPredictor, self).__init__()
        self.fc1 = nn.Linear(in_channels, mid_channels)
        self.fc2 = nn.Linear(mid_channels, mid_channels)
        self.cls_score = nn.Linear(mid_channels, num_classes)
        self.bbox_pred = nn.Linear(mid_channels, num_classes * 4)
        
    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        score = self.cls_score(x)
        bbox_delta = self.bbox_pred(x)

        return score, bbox_delta

class MaskRCNNPredictor(nn.Sequential):
    def __init__(self, in_channels, layers, dim_reduced, num_classes):
        """
        Arguments:
            in_channels (int)
            layers (Tuple[int])
            dim_reduced (int)
            num_classes (int)
        """

        d = OrderedDict()
        next_feature = in_channels
        for layer_idx, layer_features in enumerate(layers, 1):
            d['mask_fcn{}'.format(layer_idx)] = nn.Conv2d(next_feature, layer_features, 3, 1, 1)
            d['relu{}'.format(layer_idx)] = nn.ReLU(inplace=True)
            next_feature = layer_features

        ### output is [14x14] -> [28x28]
        ### nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        d['mask_conv5'] = nn.ConvTranspose2d(next_feature, dim_reduced, 2, 2, 0)
        d['relu5'] = nn.ReLU(inplace=True)
        ### output is [28x28] -> [56x56]
        # d['mask_conv6'] = nn.ConvTranspose2d(next_feature, dim_reduced, 2, 2, 0)
        # d['relu6'] = nn.ReLU(inplace=True)
        # ### output is [56x56] -> [112x112]
        # d['mask_conv7'] = nn.ConvTranspose2d(next_feature, dim_reduced, 2, 2, 0)
        # d['relu7'] = nn.ReLU(inplace=True)
        # ## output is [112x112] -> [224x224]
        # d['mask_conv8'] = nn.ConvTranspose2d(next_feature, dim_reduced, 2, 2, 0)
        # d['relu8'] = nn.ReLU(inplace=True)

        d['mask_fcn_logits'] = nn.Conv2d(dim_reduced, num_classes, 1, 1, 0)
        super(MaskRCNNPredictor, self).__init__(d)

        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')

###############################
###############################
    
def ResNetMaskRCNN(pretrained=config.IS_PRETRAINED, pretrained_backbone=True,
                   backbone_feat_extractor=config.BACKBONE_FEAT_EXTRACTOR,
                   num_classes=config.NUM_CLASSES):
    """
    Constructs a Mask R-CNN model with a ResNet-50 backbone.
    
    Arguments:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017.
        num_classes (int): number of classes (including the background).
    """
    # print()

    ##############################################
    ### TODO: backbone
    ##############################################

    if pretrained:
        pretrained_backbone = False

    backbone = ResNetBackbone(backbone_name=backbone_feat_extractor, pretrained=True)

    # import torchvision.models as models
    # print('using pretrained VGG16 weights ..')
    # backbone = models.vgg16(pretrained=True)
    # ### print(backbone)
    # backbone = nn.Sequential(*list(backbone.features.children())[:-1])
    # print(backbone)

    # import torchvision.models as models
    # print('using pretrained ResNet18 weights ..')
    # backbone = models.resnet18(pretrained=True)
    # ### print(backbone)
    # backbone = list(backbone.children())[:-2]
    # backbone = nn.Sequential(*backbone)
    ### print(backbone)

    # from torchsummary import summary
    # TORCH_SUMMARY = (config.NUM_CHANNELS, config.INPUT_SIZE[0], config.INPUT_SIZE[1])
    # summary(backbone.to(config.DEVICE), TORCH_SUMMARY)

    # backbone = ResNet101(nInputChannels=config.NUM_CHANNELS, os=config.OUTPUT_STRIDE, pretrained=pretrained_backbone)

    ##############################################
    ##############################################

    model = MaskRCNN(backbone, num_classes)

    ##############################################
    ### TODO: look at loading pre-trained weights
    ##############################################
    
    if pretrained:
        # if pretrained_backbone:
        #     print("loading pre-trained ResNet weights ..")
        # print("loading pre-trained MaskRCNN weights: {} .. ".format(config.MASKRCNN_PRETRAINED_WEIGHTS))
        # print('num classes: {} ..'.format(num_classes))

        checkpoint = torch.load(config.MASKRCNN_PRETRAINED_WEIGHTS, map_location=config.DEVICE)
        pretrained_msd = checkpoint["model"]
        pretrained_msd_values = list(pretrained_msd.values())
        pretrained_msd_names = list(pretrained_msd.keys())

        # We want to remove the heads
        # 36 - head.box_predictor.cls_score.weight:
        # 37 - head.box_predictor.cls_score.bias:
        # 38 - head.box_predictor.bbox_pred.weight:
        # 39 - head.box_predictor.bbox_pred.bias:
        # 50 - head.mask_predictor.mask_fcn_logits.weight:
        # 51 - head.mask_predictor.mask_fcn_logits.bias:

        msd = model.state_dict()
        msd_names = list(msd.keys())
        # skip_list = [36, 37, 38, 39, 50, 51]          # VGG16
        # skip_list = [130, 131, 132, 133, 144, 145]      # ResNet18
        skip_list = [114, 115, 116, 117, 128, 129]  # ResNet18
        for i, name in enumerate(msd):
            if i in skip_list:
                continue
            # print(f'\tmsd_names:{msd_names[i]}')
            msd[name].copy_(pretrained_msd_values[i])
        model.load_state_dict(msd)

        ##############################################
        ##############################################

        # pretrained_msd = load_url(config.MASKRCNN_PRETRAINED_WEIGHTS)
        # pretrained_msd_values = list(pretrained_msd.values())
        # pretrained_msd_names = list(pretrained_msd.keys())
        #
        # '''
        # 265 to 271: backbone.fpn weights
        # 273 to 279: backbone.fpn weights
        # '''
        #
        # del_list = [i for i in range(265, 271)] + [i for i in range(273, 279)]
        # for i, del_idx in enumerate(del_list):
        #     pretrained_msd_values.pop(del_idx - i)
        #
        # '''
        # 271: 'rpn.head.cls_logits.weight'
        # 272: 'rpn.head.cls_logits.bias'
        # 273: 'rpn.head.bbox_pred.weight'
        # 274: 'rpn.head.bbox_pred.bias'
        # 279: 'head.box_predictor.cls_score.weight'
        # 280: 'head.box_predictor.cls_score.bias'
        # 281: 'head.box_predictor.bbox_pred.weight'
        # 282: 'head.box_predictor.bbox_pred.bias'
        # 293 to end: 'head.mask_predictor'
        # '''
        #
        # msd = model.state_dict()
        # msd_names = list(msd.keys())
        # skip_list = [271, 272, 273, 274, 279, 280, 281, 282, 293, 294, 295, 296, 297, 298, 299, 300, 301]
        # if num_classes == 91:
        #     skip_list = [271, 272, 273, 274]
        # for i, name in enumerate(msd):
        #     if i in skip_list:
        #         continue
        #     # print(f'\tmsd_names:{msd_names[i]}')
        #     msd[name].copy_(pretrained_msd_values[i])
        #
        # model.load_state_dict(msd)
    
    return model