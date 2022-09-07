from functools import partial
import torch
from torch.nn import functional as F
from torchvision.ops import focal_loss
import math


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU


class Loss:
    def __init__(
            self,
            model: torch.nn.Module,
            num_classes: int = 18,
            num_layers: int = 4,
            cls_pw: float = 1.,
            obj_pw: float = 1.,
            num_coord: int = 34,
            focal_alpha: float = .25,
            focal_gamma: float = 1.5,
            anchor_threshold: float = 4.,
            bbox_loss_gain: float = .05,
            cls_loss_gain: float = .5,
            obj_loss_gain: float = 1.,
            kp_loss_gain: float = .05,
    ):
        self.model = model
        self.num_classes = num_classes
        self.anchors = [module.anchor for module in list(self.model.children())[-1]]
        self.num_anchors = len(self.anchors[0].flatten()) // 2
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.num_layers = num_layers
        self.num_coord = num_coord
        self.anchor_threshold = anchor_threshold

        self.bce_cls = partial(focal_loss.sigmoid_focal_loss, alpha=self.focal_alpha, gamma=self.focal_gamma, reduction='mean') \
            if self.focal_gamma > .0 \
            else partial(F.binary_cross_entropy_with_logits, pos_weight=torch.tensor([cls_pw], device=self.model.device))

        self.bce_obj = partial(focal_loss.sigmoid_focal_loss, alpha=self.focal_alpha, gamma=self.focal_gamma, reduction='mean') \
            if self.focal_gamma > .0 \
            else partial(F.binary_cross_entropy_with_logits,
                         pos_weight=torch.tensor([obj_pw], device=self.model.device))

        self.balance = {
            3: [4., 1., .4],
        }.get(num_layers, [4., 1., .25, .06, .02])

        self.bbox_loss_gain = bbox_loss_gain
        self.cls_loss_gain = cls_loss_gain
        self.obj_loss_gain = obj_loss_gain
        self.kp_loss_gain = kp_loss_gain

    def build_targets(self, inputs, targets):
        num_target = targets.shape[0]

        target_cls, target_box, target_kps, indices, anchors = [], [], [], [], []
        gain = torch.ones(7 + self.num_coord * 3 // 2, device=targets.device)
        ai = torch.arange(self.num_anchors, device=targets.device).float().view(self.num_anchors, 1).repeat(1, num_target)
        targets = torch.cat((
            targets.repeat(self.num_anchors, 1, 1), ai[:, :, None]
        ), dim=2)

        g = .5
        off = torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], device=targets.device).float() * g

        for i in range(self.num_layers):
            anchor = self.anchors[i].to(targets.device)
            xy_gain = torch.tensor(inputs[i].shape, device=targets.device)[[3, 2]]
            gain[2:4] = xy_gain
            gain[4:6] = xy_gain
            for j in range(self.num_coord // 2):
                kp_idx = 6 + j * 3
                gain[kp_idx: kp_idx + 2] = xy_gain

            t = targets * gain

            if num_target:
                # Matches
                r = t[:, :, 4:6] / anchor[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.anchor_threshold  # compare
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b = t[:, 0].long()  # image
            c = t[:, 1].long()  # class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            if self.num_coord:
                kp = t[:, 6:-1].reshape(-1, self.num_coord // 2, 3)
                kp[..., :2] -= gij[:, None, :]  # grid kp relative to grid box anchor
                target_kps.append(kp)

            # Append
            a = t[:, -1].long()  # anchor indices
            indices.append(
                (b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            target_box.append(torch.cat((gxy - gij, gwh), 1))  # box
            anchors.append(anchor[a])  # anchors
            target_cls.append(c)  # class
        return target_cls, target_box, target_kps, indices, anchors

    def __call__(self, inputs: torch.Tensor, targets: torch.Tensor):

        device = targets.device

        loss_cls = torch.zeros(1, device=device)
        loss_box = torch.zeros(1, device=device)
        loss_kps = torch.zeros(1, device=device)
        loss_obj = torch.zeros(1, device=device)

        target_cls, target_box, target_kps, indices, anchors = self.build_targets(inputs, targets)
        for idx, _input in enumerate(inputs):
            b, a, gj, gi = indices[idx]  # image, anchor, gridy, gridx
            target_obj = torch.zeros_like(_input[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = _input[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[idx]  # range [0, 4] * anchor
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, target_box[idx], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                loss_box += (1.0 - iou).mean()  # iou loss

                # Keypoints
                if self.num_coord:
                    tkp = target_kps[idx]
                    vis = tkp[..., 2] > 0
                    tkp_vis = tkp[vis]
                    if len(tkp_vis):
                        pkp = ps[:, 5 + self.num_classes:].reshape((-1, self.num_coord // 2, 2))
                        pkp = (pkp.sigmoid() * 4. - 2.) * anchors[idx][:, None, :]  # range [-2, 2] * anchor
                        pkp_vis = pkp[vis]
                        l2 = torch.linalg.norm(pkp_vis - tkp_vis[..., :2], dim=-1)
                        loss_kps += torch.mean(l2)

                # Object
                score_iou = iou.detach().clamp(0).type(target_obj.dtype)
                target_obj[b, a, gj, gi] = score_iou  # iou ratio

                # Classification
                if self.num_classes > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:5 + self.num_classes], 0., device=device)  # targets
                    t[range(n), target_cls[idx]] = 1.
                    loss_cls += self.bce_cls(ps[:, 5:5 + self.num_classes], t)

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.bce_obj(_input[..., 4], target_obj)
            loss_obj += obji * self.balance[idx]  # obj loss

        loss_box *= self.bbox_loss_gain
        loss_obj *= self.obj_loss_gain
        loss_cls *= self.cls_loss_gain
        loss_kps *= self.kp_loss_gain

        bs = target_obj.shape[0]  # batch size
        # loss = loss_box + loss_obj + loss_cls + loss_kps
        # return loss * bs#, torch.cat((loss_box, loss_obj, loss_cls, loss_kps)).detach()

        return {
            'loss_box': loss_box * bs,
            'loss_obj': loss_obj * bs,
            'loss_cls': loss_cls * bs,
            'loss_kps': loss_kps * bs,
        }


