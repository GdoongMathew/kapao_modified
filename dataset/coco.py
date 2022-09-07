from typing import \
    Optional

import json
import os

import numpy as np
import cv2
import torch
from torch.utils.data import \
    Dataset, \
    DataLoader

import albumentations as alb
from albumentations.pytorch import ToTensorV2

from .utils import xywhn2xyxy, xyxy2xywhn, load_image


class COCODataset(Dataset):
    def __init__(
            self,
            image_dir: str,
            label_dir: str,
            image_size: (int, int) = (640, 640),
            batch_size: int = 8,
            stride: int = 32,
            datatype: str = 'train',
            cached_label: str = None,
            mosaic: bool = False,
            augment: Optional[alb.Compose] = None,
    ):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.stride = stride
        self.mosaic = mosaic
        self.augment = augment or alb.Compose([
            alb.Resize(height=self.image_size[0],
                       width=self.image_size[1],
                       always_apply=True),
            alb.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ],
            bbox_params=alb.BboxParams(format='yolo', label_fields=['class_labels']),
        )

        assert datatype in ['train', 'val']
        self.datatype = datatype

        if cached_label:
            self.img_list, self.annotation = self._parse_cache(cache_path=cached_label, datatype=self.datatype)
        else:
            self.img_list, self.annotation = self._parse_labeling(self.datatype, self.label_dir, self.image_dir)

    @staticmethod
    def _parse_cache(
            cache_path: str,
            datatype: str
    ):
        cache = os.path.join(cache_path, f'{datatype}2017.cache')
        data = np.load(cache, allow_pickle=True).item()

        pop_keys = ['hash', 'version', 'results', 'msgs']
        for key in pop_keys:
            data.pop(key)

        img_list, annotation_list = [], []
        for key, val in data.items():
            if val[0].shape[0]:
                img_list.append(os.path.basename(key))

                if len(val[0]):
                    val[0][:, 1:5] = xywhn2xyxy(val[0][:, 1:5], val[1][0], val[1][1])
                    val[0][:, 1:5] = xyxy2xywhn(val[0][:, 1:5], w=val[1][0], h=val[1][1], clip=True, eps=1e-3)

                annotation_list.append(val)

        return img_list, annotation_list

    @staticmethod
    def _parse_labeling(
            datatype: str,
            label_dir: str,
            image_dir: str,
    ):
        label_file = os.path.join(label_dir, f'instances_{datatype}2017.json')
        with open(label_file, 'r') as file:
            data = json.load(file)

        image_dict = {img_data['id']: (img_data['file_name'], img_data['width'], img_data['height']) for img_data in data['images']}
        annotation_dict = {}

        for annotation in data['annotations']:
            if annotation['image_id'] not in annotation_dict:
                annotation_dict[annotation['image_id']] = []
            annotation_dict[annotation['image_id']].append([annotation['category_id'], *annotation['bbox']])

        img_list, annotation_list = [], []
        for key, img_data in image_dict.items():
            img, w, h = img_data
            img_path = os.path.join(image_dir, f'{datatype}2017', img)

            if os.path.exists(img_path) and key in annotation_dict:
                img_list.append(img_path)

                label = [np.array(annotation_dict[key]), (w, h), []]
                bboxes = label[0][:, 1:5]
                bboxes[:, 2:] = bboxes[:, :2] + bboxes[:, 2:]
                label[0][:, 1:5] = xyxy2xywhn(bboxes, w=label[1][0], h=label[1][1], clip=True, eps=1e-3)

                annotation_list.append(label)

        return img_list, annotation_list

    @staticmethod
    def collate_fn(batched_data):
        imgs, label = zip(*batched_data)
        batch_label = []
        for i, _label in enumerate(label):
            if not _label.shape[0]:
                continue
            _label[:, 0] = i
            batch_label.append(_label)
        imgs = torch.stack(imgs, dim=0)
        batch_label = torch.cat(batch_label, dim=0)
        return imgs, batch_label

    def _get_one(self, index: int):
        img_path = os.path.join(self.image_dir, f'{self.datatype}2017', self.img_list[index])
        ann = self.annotation[index].copy()
        img, ori_shape, _ = load_image(img_path, output_shape=None)

        if self.augment:
            ann_numpy, image_scale, _ = ann
            ann_numpy[:, 1:5] = np.clip(ann_numpy[:, 1:5], a_min=1e-10, a_max=1.)

            bboxes = ann_numpy[:, 1:]
            class_labels = ann_numpy[:, 0]

            new = self.augment(
                image=img,
                bboxes=bboxes,
                class_labels=class_labels,
            )

            img = new['image']
            ann = np.array([[c, *b] for c, b in zip(new['class_labels'], new['bboxes'])])

        if not isinstance(ann, (np.ndarray, torch.Tensor)):
            ann = np.asarray(ann[0])

        # create an empty space for indexing when batching data

        if ann.shape[0]:
            ann_out = torch.zeros((ann.shape[0], ann.shape[1] + 1))
            ann_out[:, 1:] = torch.from_numpy(ann)
            return img, ann_out
        return img, torch.from_numpy(ann)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(
            self,
            index
    ):
        if not self.mosaic:
            return self._get_one(index)

        return self._get_one(index)


if __name__ == '__main__':

    training_shape = (640, 640)
    aug = alb.Compose([
        # alb.Blur(p=.7),
        # alb.ToGray(p=.7),
        alb.Resize(width=training_shape[0], height=training_shape[1], p=1),
        # alb.ShiftScaleRotate(
        #     shift_limit=.1,
        #     scale_limit=(.5, 1.1),
        #     border_mode=cv2.BORDER_CONSTANT
        # ),
        # alb.SafeRotate(limit=20, border_mode=cv2.BORDER_CONSTANT, p=.5),
        # alb.ColorJitter(p=.5),
        # alb.MotionBlur(),
        # alb.RandomGamma(p=.3),
        # alb.ImageCompression(quality_lower=75, p=.5),
        # alb.Normalize(
        #     mean=[0.485, 0.456, 0.406],
        #     std=[0.229, 0.224, 0.225],
        # ),
        ToTensorV2(),
    ],
        bbox_params=alb.BboxParams(format='yolo', label_fields=['class_labels', ]),
    )

    coco_data = COCODataset(
        image_dir='/media/gdoongmathew/WD_Drive/Data/COCO_2017/images',
        label_dir='/media/gdoongmathew/WD_Drive/Data/COCO_2017/annotations',
        datatype='train',
        image_size=training_shape,
        augment=aug
    )

    for img, ann in coco_data:
        # img, ann = coco_data[i]
        img = img.numpy().transpose((1, 2, 0))

        ann = ann.numpy()
        for _ann in ann:
            bboxes = _ann[2:]

            center = (
                int(bboxes[0] * img.shape[1]),
                int(bboxes[1] * img.shape[0])
            )
            w, h = int(bboxes[2] * img.shape[1]), int(bboxes[3] * img.shape[0])

            img = cv2.rectangle(
                img,
                (center[0] - w // 2, center[1] - h // 2),
                (center[0] + w // 2, center[1] + h // 2),
                color=(255, 255, 3),
                thickness=2,
            )

        cv2.imshow('img', img)
        cv2.waitKey(0)
