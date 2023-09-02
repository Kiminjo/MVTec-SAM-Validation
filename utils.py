import cv2 
import numpy as np
from skimage import morphology
from typing import List

def separate_objects(mask: np.array
                     ) -> List[np.array]:
    _, binary_mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    object_masks = []

    for contour in contours:
        object_mask = np.zeros_like(mask)
        cv2.drawContours(object_mask, [contour], -1, 255, thickness=cv2.FILLED)
        object_masks.append(object_mask)

    return object_masks

def create_bbox_prompt(masks: List[np.array],
                       padding: int = 0
                       ) -> List[List[int]] :
    bboxes = []
    for mask in masks: 
        nonzero_idx = np.nonzero(mask)
        upper = np.min(nonzero_idx[0]) + padding
        lower = np.max(nonzero_idx[0]) + padding
        left = np.min(nonzero_idx[1]) + padding
        right = np.max(nonzero_idx[1]) + padding
        bboxes.append([left, upper, right, lower])
    return np.array(bboxes)

def create_point_prompt(masks: List[np.array]
                        ) -> List[List[int]] : 
    points = []
    for mask in masks:
        nonzero_idx = np.nonzero(mask)
        sampled_idx = int(len(nonzero_idx[0]) / 2)
        x, y = nonzero_idx[1][sampled_idx], nonzero_idx[0][sampled_idx]
        points.append([x, y])
    labels = [1] * len(points)    
    return np.array(points), np.array(labels)

def mask_postprocessing(mask: np.array,
                        kernel_size: int = 4
                        ) -> np.array:
    kernel = morphology.disk(kernel_size)
    mask = morphology.opening(mask, kernel)
    return mask * 255

