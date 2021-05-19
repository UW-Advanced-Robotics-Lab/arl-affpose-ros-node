import glob
import numpy as np

import rospy

import cv2
from PIL import Image
import matplotlib.pyplot as plt

#######################################
#######################################

from utils import helper_utils
from utils.dataset import vicon_dataset_utils

#######################################
#######################################

def load_obj_ply_files(CLASSES_FILE, CLASS_IDS_FILE):

    ROOT_PATH = '/home/akeaveny/catkin_ws/src/AffDenseFusionROSNode/utils/'

    ###################################
    # OG PLY
    ###################################

    class_file = open(CLASSES_FILE)
    class_id_file = open(CLASS_IDS_FILE)
    class_IDs = np.loadtxt(class_id_file, dtype=np.int32)

    class_IDs = np.array([class_IDs])

    cld = {}
    print("\n********* Loading Object Meshes ************")
    for class_id in class_IDs:
        class_input = class_file.readline()
        if not class_input:
            break
        input_file = open(ROOT_PATH + 'object_meshes/models/{0}/densefusion/{0}.xyz'.format(class_input.rstrip()))
        cld[class_id] = []
        while 1:
            input_line = input_file.readline()
            if not input_line:
                break
            input_line = input_line[:-1].split(' ')
            cld[class_id].append([float(input_line[0]), float(input_line[1]), float(input_line[2])])
        cld[class_id] = np.array(cld[class_id])
        print("class_id: {}".format(class_id))
        print("class_input: {}".format(class_input.rstrip()))
        print("Num Point Clouds: {}\n".format(len(cld[class_id])))
        input_file.close()

    ##################################
    ##################################

    class_file = open(CLASSES_FILE)
    obj_classes = np.loadtxt(class_file, dtype=np.str)

    class_id_file = open(CLASS_IDS_FILE)
    class_IDs = np.loadtxt(class_id_file, dtype=np.int32)

    class_IDs = np.array([class_IDs])
    obj_classes = np.array([obj_classes])

    #################################
    #################################

    return cld, obj_classes, class_IDs
