import os

import torch.utils.data
from kapao import KAPAO
from dataset.coco_hpe import COCOKptDataset
from pytorch_lightning.trainer import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import callbacks

import albumentations as alb
from albumentations.pytorch import ToTensorV2
import cv2

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.cuda.empty_cache()
training_shape = (512, 512)

anchors = [
    (19, 27, 44, 40, 38, 94),
    (96, 69, 86, 152, 180, 137),
    (140, 301, 303, 264, 238, 542),
    (436, 615, 739, 380, 925, 792)
]

data_workers = 6
batch_size = 20
max_epoch = 200

model = KAPAO(
    num_class=18,
    n_keypoints=17,
    n_blocks_exchange=4,
    anchors=anchors,
    max_epoch=max_epoch
)

aug = alb.Compose([
    alb.Blur(p=.5),
    alb.ToGray(p=.5),
    alb.Resize(width=training_shape[0], height=training_shape[1], p=1),
    alb.CLAHE(p=.5),
    alb.HorizontalFlip(p=.5),
    alb.ShiftScaleRotate(
        shift_limit=.05,
        scale_limit=(.5, 1.1),
        rotate_limit=0,
        border_mode=cv2.BORDER_CONSTANT
    ),
    alb.SafeRotate(
        limit=20,
        border_mode=cv2.BORDER_CONSTANT
    ),
    alb.ColorJitter(p=.5),
    alb.MotionBlur(p=.5),
    alb.RandomGamma(p=.3),
    alb.ImageCompression(quality_lower=75, p=.5),
    alb.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
    ToTensorV2(),
],
    bbox_params=alb.BboxParams(format='yolo', label_fields=['class_labels', 'bbox_idx']),
    keypoint_params=alb.KeypointParams(format='xy', remove_invisible=False)
)

train_loader = torch.utils.data.DataLoader(
    dataset=COCOKptDataset(
        datatype='train',
        image_dir='/media/gdoongmathew/WD_Drive/Data/COCO_2017/images',
        label_dir='/media/gdoongmathew/WD_Drive/Data/COCO_2017/annotations',
        cached_label='/media/gdoongmathew/WD_Drive/Data/COCO_2017/kp_labels_coco/img_txt',
        image_size=training_shape,
        augment=aug
    ),
    batch_size=batch_size,
    shuffle=True,
    collate_fn=COCOKptDataset.collate_fn,
    num_workers=data_workers,
)


val_loader = torch.utils.data.DataLoader(
    dataset=COCOKptDataset(
        datatype='val',
        image_dir='/media/gdoongmathew/WD_Drive/Data/COCO_2017/images',
        label_dir='/media/gdoongmathew/WD_Drive/Data/COCO_2017/annotations',
        cached_label='/media/gdoongmathew/WD_Drive/Data/COCO_2017/kp_labels_coco/img_txt',
        image_size=training_shape,
        # augment=aug
    ),
    shuffle=False,
    batch_size=batch_size,
    collate_fn=COCOKptDataset.collate_fn,
    num_workers=data_workers,
)

trainer = Trainer(
    precision=16,
    accelerator='gpu',
    max_epochs=max_epoch,
    logger=[
        pl_loggers.TensorBoardLogger(save_dir='./logs/'),
    ],
    callbacks=[
        callbacks.GPUStatsMonitor(temperature=True),
        callbacks.LearningRateMonitor(logging_interval='epoch', log_momentum=True),
        callbacks.ModelCheckpoint(dirpath='./logs', save_top_k=2, monitor='loss')
    ]
)

trainer.fit(
    model=model,
    train_dataloaders=train_loader,
    val_dataloaders=val_loader,
)
