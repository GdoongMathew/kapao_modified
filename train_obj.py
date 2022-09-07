import os

import torch.utils.data
from kapao import KAPAO
from dataset.coco import COCODataset
from pytorch_lightning.trainer import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import callbacks

import albumentations as alb
from albumentations.pytorch import ToTensorV2
import cv2

import mlflow.pytorch as mltorch

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.cuda.empty_cache()
training_shape = (512, 512)

anchors = [
    [10, 13, 16, 30, 33, 23],
    [30, 61, 62, 45, 59, 119],
    [116, 90, 156, 198, 373, 326],  # P5/32
    [436, 615, 739, 380, 925, 792]
]

data_workers = 6
batch_size = 32
max_epoch = 200

model = KAPAO(
        num_class=91,
        n_keypoints=0,
        n_blocks_exchange=4,
        anchors=anchors,
        max_epoch=max_epoch,
        module_ch=[64, 64, 128, 256, 384, 512],
        repeat_n=[1, 3, 3, 1, 1],
    )

aug = alb.Compose([
    alb.Blur(p=.5),
    alb.ToGray(p=.5),
    alb.Resize(width=training_shape[0], height=training_shape[1], p=1),
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
    bbox_params=alb.BboxParams(format='yolo', label_fields=['class_labels', ]),
)

train_loader = torch.utils.data.DataLoader(
    dataset=COCODataset(
        datatype='train',
        image_dir='/media/gdoongmathew/WD_Drive/Data/COCO_2017/images',
        label_dir='/media/gdoongmathew/WD_Drive/Data/COCO_2017/annotations',
        image_size=training_shape,
        augment=aug
    ),
    batch_size=batch_size,
    shuffle=True,
    collate_fn=COCODataset.collate_fn,
    num_workers=data_workers,
)


val_loader = torch.utils.data.DataLoader(
    dataset=COCODataset(
        datatype='val',
        image_dir='/media/gdoongmathew/WD_Drive/Data/COCO_2017/images',
        label_dir='/media/gdoongmathew/WD_Drive/Data/COCO_2017/annotations',
        image_size=training_shape,
        # augment=aug
    ),
    shuffle=False,
    batch_size=batch_size,
    collate_fn=COCODataset.collate_fn,
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

mltorch.autolog()

trainer.fit(
    model=model,
    train_dataloaders=train_loader,
    val_dataloaders=val_loader,
)
