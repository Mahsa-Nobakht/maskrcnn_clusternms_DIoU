import torch
from torch import triu

from detectron2.layers.losses import diou_loss


def cluster_nms(boxes, scores, lvl, iou_threshold: float):
    # scores, classes = scores.max(dim=0)
    T = 5500
    t = 1
    _, idx = scores.sort(0, descending=True)
    idx = idx[:T]
    boxes_idx = boxes[idx]
    # T = len(boxes)

    t_star = T
    b_zero = 1
    X = torch.empty(T, T)

    for i in range(T):
        for j in range(T):
            diou = diou_loss(boxes_idx[i], boxes_idx[j])
            X[i, j] = diou

    X = triu(X)  # upper triangular iou matrix
    B = X
    while t < T:
       A = B
       maxA, _ = torch.max(A, dim=0)
       # maxA = A.max(dim=0)[0]
       # E = (maxA == 0).unsqueeze(1).expand_as(A)
       E = (maxA <= iou_threshold).float().unsqueeze(1).expand_as(A)
       B = X.mul(E)
       # E = (maxA == 0).expand_as(A)
       # B = X & E
       if A.equal(B) == True:
           break
    idx_out = idx[maxA <= iou_threshold]
    bs = boxes[idx_out]
    # return bs
    return idx_out
    # x=0


#############################################################################
    # # Collapse all the classes into 1
    # scores, classes = scores.max(dim=0)
    # _, idx = scores.sort(0, descending=True)
    # idx = idx[:top_k]
    # boxes_idx = boxes[idx]
    # iou = diou_loss(boxes_idx, boxes_idx).triu_(diagonal=1)
    # B = iou
    # for i in range(200):
    #     A = B
    #     maxA, _ = torch.max(A, dim=0)
    #     E = (maxA <= iou_threshold).float().unsqueeze(1).expand_as(A)
    #     B = iou.mul(E)
    #     if A.equal(B) == True:
    #         break
    # idx_out = idx[maxA <= iou_threshold]
    # return boxes[idx_out], masks[idx_out], classes[idx_out], scores[idx_out]