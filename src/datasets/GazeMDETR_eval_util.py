""" Evaluator for GazeMDETR test sets """
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import struct

import numpy as np
from prettytable import PrettyTable

import util.dist as dist

IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640

# camera real sense d415
ICUB_CRIS_CAM_INTRINSIC = np.zeros(shape=(3, 3), dtype=np.float64)
ICUB_CRIS_CAM_INTRINSIC[0, 0] = 618.071  # fx
ICUB_CRIS_CAM_INTRINSIC[0, 2] = 305.902  # cx
ICUB_CRIS_CAM_INTRINSIC[1, 1] = 617.783  # fy
ICUB_CRIS_CAM_INTRINSIC[1, 2] = 246.352  # cy
ICUB_CRIS_CAM_INTRINSIC[2, 2] = 1.0

# #### Bounding box utilities imported from torchvision and converted to numpy
# def box_area(boxes):
#     return (boxes[2] - boxes[0]) * (boxes[3] - boxes[1])

# # IoU calculation
# def _box_inter_union(boxes1: np.array, boxes2: np.array) -> Tuple[np.array, np.array]:
#     area1 = box_area(boxes1)
#     area2 = box_area(boxes2)

#     lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
#     rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

#     wh = (rb - lt).clip(min=0)  # [N,M,2]
#     inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

#     union = area1[:, None] + area2 - inter

#     return inter, union


# def box_iou(boxes1: np.array, boxes2: np.array) -> np.array:
#     """
#     Return intersection-over-union (Jaccard index) of boxes.

#     Both sets of boxes are expected to be in ``(x1, y1, x2, y2)`` format with
#     ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

#     Args:
#         boxes1 (Tensor[N, 4])
#         boxes2 (Tensor[M, 4])

#     Returns:
#         iou (Tensor[N, M]): the NxM matrix containing the pairwise IoU values for every element in boxes1 and boxes2
#     """
#     inter, union = _box_inter_union(boxes1, boxes2)
#     iou = inter / union
#     return iou

def bbox_center(bbox_coordinates):
    """Computes the center of the boundingbox

    Args:
        bbox_coordinates (list): Coordinates of the boundingbox in the form of [Xmin, Ymin, Xmax, Ymax]
    return: 
        bbox_center (list): Center point of the boundingbox [X, Y]
    """
    x, y = (bbox_coordinates[:, 0] + bbox_coordinates[:, 2])/2 , (bbox_coordinates[:, 1] + bbox_coordinates[:, 3])/2
    return (x, y)

def distance_pixels(bbox_1, bbox_2):
    """Computes the distance between the centers of 2 bounding boxes (Evaluated in pixels).

    Args:
        bbox_1 (list): Coordinates of the boundingbox 1 in the form of [Xmin, Ymin, Xmax, Ymax]
        bbox_2 (list): Coordinates of the boundingbox 2 in the form of [Xmin, Ymin, Xmax, Ymax]

    Returns:
        float: the distance between 2 boundingboxes in 2D  
    """
    
    bbox1_center = bbox_center(bbox_1)
    bbox2_center = bbox_center(bbox_2)
    
    distance = np.linalg.norm(np.array(bbox1_center)- np.array(bbox2_center))
    
    return distance

def read_depth(filename):
    with open(filename, 'rb') as f:
        height = f.read(8)
        height = int.from_bytes(height, "little")
        assert height == IMAGE_HEIGHT
        
        width = f.read(8)
        width = int.from_bytes(width, "little")
        assert width == IMAGE_WIDTH

        depth_img = []
        while (True):
            depthval_b = f.read(4)      # binary, little endian
            if not depthval_b:
                break
            depthval_m = struct.unpack("<f", depthval_b)    # depth val as meters
            depth_img.append(depthval_m)
        assert len(depth_img) == height * width

    depth_img = np.array(depth_img, dtype=np.float32).reshape(height, width)

    return depth_img

def get_mean_depth_over_area(depth_img, pixel, range):

    vertical_range = np.zeros(2)
    vertical_range[0] = pixel[1] - round(range/2) if pixel[1] - round(range/2) > 0 else 0
    vertical_range[1] = pixel[1] + round(range/2) if pixel[1] + round(range/2) < IMAGE_HEIGHT else IMAGE_HEIGHT

    horizontal_range = np.zeros(2)
    horizontal_range[0] = pixel[0] - round(range/2) if pixel[0] - round(range/2) > 0 else 0
    horizontal_range[1] = pixel[0] + round(range/2) if pixel[0] + round(range/2) < IMAGE_WIDTH else IMAGE_WIDTH

    vertical_range = vertical_range.astype(int)
    horizontal_range = horizontal_range.astype(int)

    depth = []
    for hpix in np.arange(horizontal_range[0], horizontal_range[1]):
        for vpix in np.arange(vertical_range[0], vertical_range[1]):
            depth.append(depth_img[vpix, hpix])

    mean_depth = np.mean(depth)

    return mean_depth

def from_pixels_to_ccs(point_pixels, depth, cam_intrinsic):
    point_ccs = np.zeros(3)

    point_ccs[2] = depth                                                                     # z
    point_ccs[0] = (point_pixels[0] - cam_intrinsic[0, 2])*point_ccs[2]/cam_intrinsic[0, 0]  # x
    point_ccs[1] = (point_pixels[1] - cam_intrinsic[1, 2])*point_ccs[2]/cam_intrinsic[1, 1]  # y

    return point_ccs

def distance_meters(bbox_1, bbox_2, depth_info, cam_intrinsics):
    
    bbox1_center = bbox_center(bbox_1)
    bbox2_center = bbox_center(bbox_2)
    
    depth_image = read_depth(depth_info)
    
    bbox1_center_css = from_pixels_to_ccs(bbox1_center, get_mean_depth_over_area(depth_image, bbox1_center, 20), cam_intrinsics)
    bbox2_center_css = from_pixels_to_ccs(bbox2_center, get_mean_depth_over_area(depth_image, bbox2_center, 20), cam_intrinsics)

    distance = np.linalg.norm(bbox1_center_css, bbox2_center_css)
    
    return distance
    
    
        
        
        