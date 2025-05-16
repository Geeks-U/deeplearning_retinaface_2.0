import torch
import torch.nn as nn
import copy

from src.nets.backbone import MobileNetV2
from src.nets.fpn import FPN
from src.nets.ssh import SSH
from src.nets.head import ShareHead

cfg_default = {}

class Retinaface(nn.Module):
    def __init__(self, cfg_model=None):
        super().__init__()
        self.cfg = copy.deepcopy(cfg_default)
        if cfg_model is not None:
            self.cfg.update(cfg_model)

        self.backbone = MobileNetV2()
        self.fpn = FPN()
        self.ssh = SSH()
        self.head = ShareHead()

    def forward(self, x):
        output_backbone = self.backbone(x)
        output_fpn = self.fpn(output_backbone)
        output_ssh = self.ssh(output_fpn)
        output_head = self.head(output_ssh)

        return output_head

if __name__ == '__main__':
    model = Retinaface()
    model.eval()

    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        outputs = model(x)
    print("model模型配置:", model.cfg)

    print("各输出特征图形状：")
    for i, name in enumerate(['bbox', 'cls', 'ldm']):
        feat = outputs[name]
        print(f"{name}: {feat.shape}")
