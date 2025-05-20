import copy
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from pathlib import Path
from datetime import datetime

from src.nets.retinaface import Retinaface
from src.utils.anchor import CustomAnchors
from src.utils.loss import CustomLoss

from src.data.datamodule import DataModule  # 请替换成你的实际文件名和路径

# 默认配置
cfg_trainer_default = {
    'num_epochs': 10,
    'batch_size': 32,
    'learning_rate': 1e-2,
    'cuda': True,
    'weights_save_dir': r'D:\Code\DL\Pytorch\retinaface\weights',
    'weights_save_filename_suffix': datetime.now().strftime('%Y%m%d_%H%M%S')
}


class Trainer:
    def __init__(self, cfg_trainer=None):
        self.cfg = copy.deepcopy(cfg_trainer_default)
        if cfg_trainer is not None:
            self.cfg.update(cfg_trainer)

        self.device = 'cuda' if self.cfg['cuda'] and torch.cuda.is_available() else 'cpu'
        self.num_epochs = self.cfg['num_epochs']
        self.batch_size = self.cfg['batch_size']
        self.learning_rate = self.cfg['learning_rate']
        self.weights_save_dir = Path(self.cfg['weights_save_dir'])
        self.timestamp = self.cfg['weights_save_filename_suffix']
        self.input_image_size = self.cfg['input_image_size']

        self.weights_save_dir.mkdir(parents=True, exist_ok=True)

        self.model = Retinaface().to(self.device)
        self.anchors = CustomAnchors(cfg_anchor={'input_image_size': self.input_image_size}).get_center_anchors().to(self.device)
        print("锚框形状: ", self.anchors.size())

        self.criterion = CustomLoss()
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=0.9,
            weight_decay=5e-4
        )
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

        # 使用 Lightning DataModule 来加载数据
        self.datamodule = DataModule()
        self.datamodule.setup(stage='fit', cfg_fit={'input_image_size': self.input_image_size})  # 生成数据集和拆分
        self.train_loader = self.datamodule.train_dataloader()
        self.val_loader = self.datamodule.val_dataloader()

        self.best_loss = float('inf')

    def save_model(self, path: Path):
        torch.save(self.model.state_dict(), path)
        print(f"Saved model to {path}")

    def validate(self):
        self.model.eval()
        val_loss, val_loc, val_conf, val_landm = 0, 0, 0, 0
        with torch.no_grad():
            for images, targets in self.val_loader:
                images = torch.from_numpy(images).float().to(self.device)
                targets = [torch.from_numpy(ann).float().to(self.device) for ann in targets]

                out = self.model(images)
                loss_l, loss_c, loss_landm = self.criterion(
                    [out['bbox'], out['cls'], out['ldm']], self.anchors, targets)
                loss = 2 * loss_l + loss_c + loss_landm

                val_loss += loss.item()
                val_loc += loss_l.item()
                val_conf += loss_c.item()
                val_landm += loss_landm.item()

        avg_val_loss = val_loss / len(self.val_loader)
        avg_val_loc = val_loc / len(self.val_loader)
        avg_val_conf = val_conf / len(self.val_loader)
        avg_val_landm = val_landm / len(self.val_loader)

        print(f"[Validation] Avg Loss: {avg_val_loss:.4f} "
              f"(Loc: {avg_val_loc:.4f}, Conf: {avg_val_conf:.4f}, Landm: {avg_val_landm:.4f})")

        return avg_val_loss

    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss, total_loc, total_conf, total_landm = 0, 0, 0, 0

            for i, (images, targets) in enumerate(self.train_loader):
                images = torch.from_numpy(images).float().to(self.device)
                targets = [torch.from_numpy(ann).float().to(self.device) for ann in targets]

                self.optimizer.zero_grad()

                out = self.model(images)
                loss_l, loss_c, loss_landm = self.criterion(
                    [out['bbox'], out['cls'], out['ldm']], self.anchors, targets)
                loss = 2 * loss_l + loss_c + loss_landm

                if torch.isinf(loss):
                    print(f"[Warning] Loss is inf at Epoch {epoch + 1}, Step {i + 1}. Skipping backward and optimizer step.")
                    continue

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                total_loc += loss_l.item()
                total_conf += loss_c.item()
                total_landm += loss_landm.item()

                if (i + 1) % 10 == 0:
                    print(f"Epoch [{epoch + 1}/{self.num_epochs}], Step [{i + 1}/{len(self.train_loader)}], "
                          f"Loss: {loss.item():.4f}, Loc: {loss_l.item():.4f}, "
                          f"Conf: {loss_c.item():.4f}, Landm: {loss_landm.item():.4f}")

            self.scheduler.step()

            avg_loss = total_loss / len(self.train_loader)
            avg_loc = total_loc / len(self.train_loader)
            avg_conf = total_conf / len(self.train_loader)
            avg_landm = total_landm / len(self.train_loader)

            print(f"[Epoch {epoch + 1}] Train - Avg Loss: {avg_loss:.4f} "
                  f"(Loc: {avg_loc:.4f}, Conf: {avg_conf:.4f}, Landm: {avg_landm:.4f})")

            avg_val_loss = self.validate()

            last_path = self.weights_save_dir / f'model_{self.timestamp}_last.pth'
            self.save_model(last_path)

            if avg_val_loss < self.best_loss:
                print('-----------------------------------------------------')
                self.best_loss = avg_val_loss
                best_path = self.weights_save_dir / f'model_{self.timestamp}_best.pth'
                self.save_model(best_path)


if __name__ == '__main__':
    cfg_trainer = {
        'num_epochs': 10,
        'batch_size': 32,
        'learning_rate': 1e-2,
        'cuda': True,
    }

    trainer = Trainer(cfg_trainer=cfg_trainer)
    trainer.train()
