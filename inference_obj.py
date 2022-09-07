from multiprocessing import freeze_support
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import albumentations as alb
from albumentations.pytorch import ToTensorV2
import cv2

import torch.utils.data
from kapao import KAPAO
from dataset.post_process import non_max_suppression
from dataset.utils import scale_coords, xyxy2xywh


if __name__ == '__main__':
    # freeze_support()

    max_epoch = 200
    input_size = 512

    anchors = [
        [10, 13, 16, 30, 33, 23],
        [30, 61, 62, 45, 59, 119],
        [116, 90, 156, 198, 373, 326],  # P5/32
        [436, 615, 739, 380, 925, 792]
    ]

    aug = alb.Compose([
        alb.Resize(height=input_size,
                   width=input_size,
                   always_apply=True),
        alb.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])

    model = KAPAO.load_from_checkpoint(
        'logs/epoch=1-step=7330.ckpt',
        anchors=anchors,
        module_ch=[64, 64, 128, 256, 384, 512],
        repeat_n=[1, 3, 3, 1, 1],
        num_class=91,
        n_keypoints=0,
        n_blocks_exchange=4,
        max_epoch=max_epoch,
    )
    # model = KAPAO(
    #     anchors=anchors,
    #     module_ch=[64, 128, 256, 384, 512, 768],
    #     num_class=91,
    #     n_keypoints=0,
    #     n_blocks_exchange=4,
    #     max_epoch=max_epoch,
    # )

    model.eval()

    image = cv2.imread('/media/gdoongmathew/WD_Drive/Data/COCO_2017/images/train2017/000000000025.jpg')

    image_torch = aug(image=image)['image']
    _, bbox = model(image_torch[None, ...])

    res = non_max_suppression(
        bbox,
        conf_thresh=.1,
        iou_thresh=.5,
        max_detection=500
    )

    for i, det in enumerate(res):
        if len(det):
            det[:, :4] = scale_coords(image_torch.shape[1:], det[:, :4], image.shape).round()
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)
                if conf < .2:
                    continue
                p1, p2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                cv2.rectangle(image, p1, p2, (255, 255, 2), thickness=4)

    cv2.imshow('img', image)
    cv2.waitKey(0)