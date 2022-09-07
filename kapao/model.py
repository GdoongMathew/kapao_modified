from typing import List

from torch.optim import lr_scheduler
from torch import nn
import torch
import pytorch_lightning as pl

from .conv_module import *
from .backbone import YoloV56

from kapao.utils.loss import Loss


class KAPAO(pl.LightningModule):
    def __init__(
            self,
            num_class: int = 18,
            n_keypoints: int = 17,
            anchors: list = [],
            inplace=True,
            n_blocks_exchange: int = 4,
            module_ch: List[int] = [64, 128, 256, 512, 768, 1024],
            repeat_n: List[int] = [3, 9, 9, 3, 3],
            lr: float = 1e-3,
            max_epoch: int = 200
    ):
        super(KAPAO, self).__init__()

        self.lr = lr
        self.max_epoch = max_epoch

        self.num_class = num_class
        self.n_keypoints = n_keypoints
        self.inplace = inplace
        self.n_blocks_exchange = n_blocks_exchange
        self.backbone = YoloV56(module_ch=module_ch, repeat_n=repeat_n)

        self.panet = PANet(
            channels=[
                *module_ch[-n_blocks_exchange:][::-1],
                *module_ch[-(n_blocks_exchange - 1):]
            ]
        )

        strides = [4 * 2 ** i for i in range(len(anchors))]

        self.heads = nn.ModuleList([
            KAPAOHead(
                anchor=anchor,
                stride=stride,
                num_coord=n_keypoints * 2,
                num_class=num_class,
                ch_input=ch,
            ) for stride, anchor, ch in zip(strides, anchors[::-1], module_ch[-self.n_blocks_exchange:])
        ])

        self.loss = Loss(self, num_classes=num_class, num_layers=len(anchors), num_coord=n_keypoints * 2)

    def _before_head_forward(self, x: torch.Tensor):
        x = self.backbone(x)
        x = self.panet(x[-self.n_blocks_exchange:][::-1])
        return x

    def forward(self, x: torch.Tensor):
        x = self._before_head_forward(x)
        assert len(x) == self.n_blocks_exchange
        x = [head(_x) for _x, head in zip(x, self.heads)]
        if self.training:
            return x
        x, y = zip(*x)
        return x, torch.cat(y, dim=1)

    def training_step(self, batch, batch_idx):
        data, label = batch
        output = self(data)
        loss_dict = self.loss(inputs=output, targets=label)
        loss = 0
        for key, val in loss_dict.items():
            self.log(key, val, batch_size=len(data))
            loss += val
        self.log('loss', loss, batch_size=len(data))
        return loss

    def training_epoch_end(self, outputs) -> None:
        sch = self.lr_schedulers()
        if isinstance(sch, lr_scheduler.ReduceLROnPlateau):
            sch.step(self.trainer.callback_metrics['loss'])

    def validation_step(self, batch, batch_idx):
        data, label = batch
        x, y = self(data)
        # x, y = zip(*output)
        loss_dict = self.loss(inputs=x, targets=label)
        loss = 0
        for key, val in loss_dict.items():
            self.log(f'val_{key}', val, batch_size=len(data))
            loss += val
        self.log('val_loss', loss, batch_size=len(data))
        return loss

    def configure_optimizers(self):
        g = [], [], []
        bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)

        for v in self.modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
                g[2].append(v.bias)
            if isinstance(v, bn):
                g[1].append(v.weight)
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
                g[0].append(v.weight)

        optimizer = torch.optim.SGD(
            g[2], lr=self.lr, momentum=.937, nesterov=True
        )

        optimizer.add_param_group({'params': g[0], 'weight_decay': 5e-4})
        optimizer.add_param_group({'params': g[1]})

        lr_sch = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=.5,
            patience=10,
            cooldown=2,
            min_lr=1e-10,
            verbose=True,
        )

        return [optimizer, ], {'scheduler': lr_sch, 'monitor': 'loss_kps'}
