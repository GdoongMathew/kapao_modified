import time
import torch
import torchvision.ops
from torchvision.ops import nms

from .utils import xywh2xyxy


def non_max_suppression(
        prediction,
        conf_thresh: float = .25,
        iou_thresh: float = .45,
        classes=None,
        labels=(),
        max_detection: int = 300,
):
    bs = prediction.shape[0]
    nc = prediction.shape[2] - 5
    cand = prediction[..., 4] > conf_thresh

    max_wh = 7680
    max_nms = 30000

    output = [torch.zeros((0, 0), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):

        x = x[cand[xi]]
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]
            v[:, 4] = 1.
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.
            x = torch.cat((x, v), dim=0)

        if not x.shape[0]:
            continue

        x[:, 5:] *= x[:, 4:5]
        box = xywh2xyxy(x[:, :4])

        conf, j = x[:, 5:].max(1, keepdim=True)
        x = torch.cat((box, conf, j.float()), dim=1)[conf.view(-1) > conf_thresh]

        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        num_bbox = x.shape[0]
        if not num_bbox:
            continue

        elif num_bbox > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        cls = x[:, 5:6] * max_wh
        boxes, scores = x[:, :4] + cls, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thresh)
        if i.shape[0] > max_detection:
            i = i[:max_detection]

        output[xi] = x[i]
    return output


def non_max_suppression_kp(
        prediction,
        conf_thresh=0.25,
        iou_thresh=0.45,
        classes=None,
        max_det=300,
        num_coords=34
):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls, keypoints]
    """

    nc = prediction.shape[2] - 5 - num_coords  # number of classes
    xc = prediction[..., 4] > conf_thresh  # candidates

    # Checks
    assert 0 <= conf_thresh <= 1, f'Invalid Confidence threshold {conf_thresh}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thresh <= 1, f'Invalid IoU {iou_thresh}, valid values are between 0.0 and 1.0'

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6 + num_coords), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:-num_coords] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        conf, j = x[:, 5:-num_coords].max(1, keepdim=True)
        kp = x[:, -num_coords:]
        x = torch.cat((box, conf, j.float(), kp), 1)[conf.view(-1) > iou_thresh]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * max_wh  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = nms(boxes, scores, iou_thresh)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thresh  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def run_nms(
        model_out,
        num_coords: int = 17,
        iou_thresh: float = .65,
        iou_thresh_kp: float = .25,
        conf_thresh_kp: float = .2,
        conf_thresh: float = .001,
):
    if iou_thresh == iou_thresh_kp and conf_thresh_kp >= conf_thresh:
        # Combined NMS saves ~0.2 ms / image
        dets = non_max_suppression_kp(model_out, conf_thresh, iou_thresh, num_coords=num_coords)
        person_dets = [d[d[:, 5] == 0] for d in dets]
        kp_dets = [d[d[:, 4] >= conf_thresh_kp] for d in dets]
        kp_dets = [d[d[:, 5] > 0] for d in kp_dets]
    else:
        person_dets = non_max_suppression_kp(model_out, conf_thresh, iou_thresh,
                                             classes=[0],
                                             num_coords=num_coords)

        kp_dets = non_max_suppression_kp(model_out, conf_thresh_kp, iou_thresh_kp,
                                         classes=list(range(1, 1 + num_coords)),
                                         num_coords=num_coords)
    return person_dets, kp_dets