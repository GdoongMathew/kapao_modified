import os

import torch.utils.data
from kapao import KAPAO
from dataset.coco import COCODataset
from pytorch_lightning.trainer import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import callbacks

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

if __name__ == '__main__':
    torch.cuda.empty_cache()
    training_shape = (512, 512)

    data_workers = 6
    batch_size = 20
    max_epoch = 200

    anchors = [
        [10, 13, 16, 30, 33, 23],
        [30, 61, 62, 45, 59, 119],
        [116, 90, 156, 198, 373, 326],  # P5/32
        [436, 615, 739, 380, 925, 792]
    ]

    model = KAPAO(
        num_class=91,
        n_keypoints=0,
        n_blocks_exchange=4,
        anchors=anchors,
        max_epoch=max_epoch,
        module_ch=[64, 64, 128, 256, 384, 512],
        repeat_n=[1, 3, 3, 1, 1],
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
    trainer.validate(
        model=model,
        ckpt_path='logs/epoch=1-step=7330.ckpt',
        dataloaders=val_loader
    )