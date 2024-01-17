# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

import torch
from torchvision.ops import boxes as box_ops
from torchvision.ops import nms  # noqa . for compatibility
import numpy as np


# def cluster_nms(boxes, scores, iou_threshold):
#     num_boxes = boxes.shape[0]
#     keep = [1] * num_boxes

#     iou_matrix = torch.zeros((num_boxes, num_boxes))
#     for i in range(num_boxes):
#         for j in range(i + 1, num_boxes):
#             iou = calculate_iou(boxes[i], boxes[j])
#             iou_matrix[i][j] = iou

#     iou_matrix = torch.triu(iou_matrix, diagonal=1)

#     for t in range(num_boxes):
#         At = torch.diag(torch.tensor(keep))
#         Ct = At.mm(iou_matrix)

#         g, _ = Ct.max(dim=1)
#         bt = (g < iou_threshold).nonzero().squeeze()

#         if torch.equal(bt, torch.tensor(keep)):
#             t_star = t
#             break

#         keep = bt

#     return keep

# def calculate_iou(box1, box2):
#     # Get the coordinates of the intersection rectangle
#     x1 = max(box1[0], box2[0])
#     y1 = max(box1[1], box2[1])
#     x2 = min(box1[2], box2[2])
#     y2 = min(box1[3], box2[3])

#     # Calculate the area of intersection rectangle
#     intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

#     # Calculate the area of both bounding boxes
#     box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
#     box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

#     # Calculate IOU
#     iou = intersection_area / float(box1_area + box2_area - intersection_area)

#     return iou




# def cluster_nms(boxes, scores, eps):
  
#     num_boxes = len(boxes)
#     t = 1
#     T = num_boxes
#     bt = np.ones(num_boxes) 

#     ious = np.zeros((num_boxes, num_boxes))
#     for i in range(num_boxes):
#         for j in range(i + 1, num_boxes):
#             ious[i, j] = calculate_iou(boxes[i], boxes[j])

#     ious = np.triu(ious, k=1) 

#     while t <= T:
#         At = np.diag(bt)
#         Ct = At.dot(ious)

#         g = np.max(Ct, axis=1)
#         bt_new = np.where(g < eps, 0, bt)

#         if np.all(bt_new == bt):
#             t_star = t
#             break

#         bt = bt_new
#         t += 1

#     return bt_star

# def calculate_iou(box1, box2):

#     x1, y1, x2, y2 = box1
#     x3, y3, x4, y4 = box2

#     xi1 = max(x1, x3)
#     yi1 = max(y1, y3)
#     xi2 = min(x2, x4)
#     yi2 = min(y2, y4)

#     inter_width = max(0, xi2 - xi1)
#     inter_height = max(0, yi2 - yi1)

#     area_inter = inter_width * inter_height
#     area_box1 = (x2 - x1) * (y2 - y1)
#     area_box2 = (x4 - x3) * (y4 - y3)

#     iou = area_inter / (area_box1 + area_box2 - area_inter)
#     return iou

# def cluster_nms(boxes, scores, eps):
#     num_boxes = len(boxes)
#     bt = torch.ones(num_boxes)

#     # Calculate IOU matrix using vectorized operations
#     x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
#     x1 = torch.maximum(x1[:, None], x1)
#     y1 = torch.maximum(y1[:, None], y1)
#     x2 = torch.minimum(x2[:, None], x2)
#     y2 = torch.minimum(y2[:, None], y2)

#     # inter_width = torch.maximum(0, x2 - x1)
#     inter_width = torch.max(torch.zeros_like(x2), x2 - x1)
#     inter_height = torch.max(torch.zeros_like(x2), y2 - y1)
#     # inter_height = torch.max(0, y2 - y1)

#     area_inter = inter_width * inter_height
#     area_box1 = (x2 - x1) * (y2 - y1)
#     area_box2 = (x2[:, None] - x1[:, None]) * (y2[:, None] - y1[:, None])

#     ious = area_inter / (area_box1 + area_box2 - area_inter)

#     # Upper triangle of the IOU matrix
#     ious = torch.triu(ious, diagonal=1)

#     while True:
#         At = torch.diag(bt)
#         Ct = At.mm(ious)

#         g = torch.max(Ct, dim=1).values
#         bt_new = torch.where(g < eps, torch.zeros_like(bt), bt)

#         if torch.all(bt_new == bt):
#             break

#         bt = bt_new

#     return bt



def cluster_nms(boxes, scores, eps):
    num_boxes = len(boxes)
    bt = torch.ones(num_boxes, device=boxes.device)

    # Calculate IOU matrix using vectorized operations
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    while True:
        At = torch.diag(bt)
        Ct = At @ boxes @ boxes.t()

        g = Ct.max(dim=1).values
        bt_new = (g >= eps).float()

        if torch.all(bt_new == bt):
            break

        bt = bt_new

    return bt

# def cluster_nms(boxes, scores, eps):
#     num_boxes = len(boxes)
#     bt = torch.ones(num_boxes)

#     # Calculate IOU matrix using vectorized operations
#     x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
#     x1 = torch.maximum(x1[:, None], x1)
#     y1 = torch.maximum(y1[:, None], y1)
#     x2 = torch.minimum(x2[:, None], x2)
#     y2 = torch.minimum(y2[:, None], y2)

#     # Intersected width and height
#     inter_width = torch.max(torch.zeros_like(x2), x2 - x1)
#     inter_height = torch.max(torch.zeros_like(x2), y2 - y1)

#     area_inter = inter_width * inter_height
#     area_box1 = (x2 - x1) * (y2 - y1)
#     area_box2 = (x2[:, None] - x1[:, None]) * (y2[:, None] - y1[:, None])

#     ious = area_inter / (area_box1 + area_box2 - area_inter)

#     # Upper triangle of the IOU matrix
#     ious = torch.triu(ious, diagonal=1)

#     while True:
#         At = torch.diag(bt)
#         Ct = At.mm(ious)

#         g = torch.max(Ct, dim=1).values
#         bt_new = torch.where(g < eps, torch.zeros_like(bt), bt)

#         if torch.all(bt_new == bt):
#             break

#         bt = bt_new

#     return bt


def batched_nms(
    boxes: torch.Tensor, scores: torch.Tensor, idxs: torch.Tensor, iou_threshold: float
):
    """
    Same as torchvision.ops.boxes.batched_nms, but with float().
    """
    assert boxes.shape[-1] == 4
    # Note: Torchvision already has a strategy (https://github.com/pytorch/vision/issues/1311)
    # to decide whether to use coordinate trick or for loop to implement batched_nms. So we
    # just call it directly.
    # Fp16 does not have enough range for batched NMS, so adding float().
    return box_ops.batched_nms(boxes.float(), scores, idxs, iou_threshold)


# Note: this function (nms_rotated) might be moved into
# torchvision/ops/boxes.py in the future
def nms_rotated(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float):
    """
    Performs non-maximum suppression (NMS) on the rotated boxes according
    to their intersection-over-union (IoU).

    Rotated NMS iteratively removes lower scoring rotated boxes which have an
    IoU greater than iou_threshold with another (higher scoring) rotated box.

    Note that RotatedBox (5, 3, 4, 2, -90) covers exactly the same region as
    RotatedBox (5, 3, 4, 2, 90) does, and their IoU will be 1. However, they
    can be representing completely different objects in certain tasks, e.g., OCR.

    As for the question of whether rotated-NMS should treat them as faraway boxes
    even though their IOU is 1, it depends on the application and/or ground truth annotation.

    As an extreme example, consider a single character v and the square box around it.

    If the angle is 0 degree, the object (text) would be read as 'v';

    If the angle is 90 degrees, the object (text) would become '>';

    If the angle is 180 degrees, the object (text) would become '^';

    If the angle is 270/-90 degrees, the object (text) would become '<'

    All of these cases have IoU of 1 to each other, and rotated NMS that only
    uses IoU as criterion would only keep one of them with the highest score -
    which, practically, still makes sense in most cases because typically
    only one of theses orientations is the correct one. Also, it does not matter
    as much if the box is only used to classify the object (instead of transcribing
    them with a sequential OCR recognition model) later.

    On the other hand, when we use IoU to filter proposals that are close to the
    ground truth during training, we should definitely take the angle into account if
    we know the ground truth is labeled with the strictly correct orientation (as in,
    upside-down words are annotated with -180 degrees even though they can be covered
    with a 0/90/-90 degree box, etc.)cluster_nms

    The way the original dataset is annotated also matters. For example, if the dataset
    is a 4-point polygon dataset that does not enforce ordering of vertices/orientation,
    we can estimate a minimum rotated bounding box to this polygon, but there's no way
    we can tell the correct angle with 100% confidence (as shown above, there could be 4 different
    rotated boxes, with angles differed by 90 degrees to each other, covering the exactly
    same region). In that case we have to just use IoU to determine the box
    proximity (as many detection benchmarks (even for text) do) unless there're other
    assumptions we can make (like width is always larger than height, or the object is not
    rotated by more than 90 degrees CCW/CW, etc.)

    In summary, not considering angles in rotated NMS seems to be a good option for now,
    but we should be aware of its implications.

    Args:
        boxes (Tensor[N, 5]): Rotated boxes to perform NMS on. They are expected to be in
           (x_center, y_center, width, height, angle_degrees) format.
        scores (Tensor[N]): Scores for each one of the rotated boxes
        iou_threshold (float): Discards all overlapping rotated boxes with IoU < iou_threshold

    Returns:
        keep (Tensor): int64 tensor with the indices of the elements that have been kept
        by Rotated NMS, sorted in decreasing order of scores
    """
    return torch.ops.detectron2.nms_rotated(boxes, scores, iou_threshold)


# Note: this function (batched_nms_rotated) might be moved into
# torchvision/ops/boxes.py in the future


@torch.jit.script_if_tracing
def batched_nms_rotated(
    boxes: torch.Tensor, scores: torch.Tensor, idxs: torch.Tensor, iou_threshold: float
):
    """
    Performs non-maximum suppression in a batched fashion.

    Each index value correspond to a category, and NMS
    will not be applied between elements of different categories.

    Args:
        boxes (Tensor[N, 5]):
           boxes where NMS will be performed. They
           are expected to be in (x_ctr, y_ctr, width, height, angle_degrees) format
        scores (Tensor[N]):
           scores for each one of the boxes
        idxs (Tensor[N]):
           indices of the categories for each one of the boxes.
        iou_threshold (float):
           discards all overlapping boxes
           with IoU < iou_threshold

    Returns:
        Tensor:
            int64 tensor with the indices of the elements that have been kept
            by NMS, sorted in decreasing order of scores
    """
    assert boxes.shape[-1] == 5

    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    boxes = boxes.float()  # fp16 does not have enough range for batched NMS
    # Strategy: in order to perform NMS independently per class,
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap

    # Note that batched_nms in torchvision/ops/boxes.py only uses max_coordinate,
    # which won't handle negative coordinates correctly.
    # Here by using min_coordinate we can make sure the negative coordinates are
    # correctly handled.
    max_coordinate = (
        torch.max(boxes[:, 0], boxes[:, 1]) + torch.max(boxes[:, 2], boxes[:, 3]) / 2
    ).max()
    min_coordinate = (
        torch.min(boxes[:, 0], boxes[:, 1]) - torch.max(boxes[:, 2], boxes[:, 3]) / 2
    ).min()
    offsets = idxs.to(boxes) * (max_coordinate - min_coordinate + 1)
    boxes_for_nms = boxes.clone()  # avoid modifying the original values in boxes
    boxes_for_nms[:, :2] += offsets[:, None]
    keep = nms_rotated(boxes_for_nms, scores, iou_threshold)
    return keep
