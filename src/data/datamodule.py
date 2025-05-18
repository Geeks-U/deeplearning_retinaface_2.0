import copy
from pathlib import Path
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

from src.data.dataset import CustomDataset, detection_collate

cfg_datamodule_default = {
    'val_split': 0.1,
    'batch_size': 32,
    'num_workers': 4,
    'pin_memory': True
}

class DataModule(pl.LightningDataModule):
    def __init__(self, cfg_datamodule=None):
        super().__init__()
        self.cfg = copy.deepcopy(cfg_datamodule_default)
        if cfg_datamodule is not None:
            self.cfg.update(cfg_datamodule)
        self.val_split = self.cfg['val_split']
        self.batch_size = self.cfg['batch_size']
        self.num_workers = self.cfg['num_workers']
        self.pin_memory = self.cfg['pin_memory']

    def prepare_data(self):
        # 数据集已经存在，无需下载
        pass

    def setup(self, stage=None, cfg_fit=None, cfg_test=None):
        if stage in (None, 'fit'):
            # 训练数据（启用 augment）
            cfg_dataset = {'augment': True}
            cfg_dataset.update(cfg_fit)
            full_dataset = CustomDataset(cfg_dataset=cfg_dataset)

            # 拆分训练/验证
            val_size = int(len(full_dataset) * self.val_split)
            train_size = len(full_dataset) - val_size
            train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

            # 验证数据禁用增强
            if isinstance(val_dataset, (list, tuple)) or hasattr(val_dataset, 'dataset'):
                if hasattr(val_dataset.dataset, 'cfg'):
                    val_dataset.dataset.cfg['augment'] = False

            self.train_dataset = train_dataset
            self.val_dataset = val_dataset

        if stage in (None, 'test'):
            # 测试数据（禁用 augment）
            cfg_dataset = {'augment': False}
            cfg_dataset.update(cfg_test)
            self.test_dataset = CustomDataset(cfg_dataset=cfg_dataset)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=detection_collate
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=detection_collate
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=detection_collate
        )
