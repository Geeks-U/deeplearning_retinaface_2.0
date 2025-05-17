import copy

import torch
import torch.nn as nn

from src.nets.backbone import build_backbone
from src.nets.fpn import build_fpn
from src.nets.ssh import build_ssh
from src.nets.head import build_head

cfg_default = {}

class Retinaface(nn.Module):
    def __init__(self, cfg_model=None):
        super().__init__()
        self.cfg = copy.deepcopy(cfg_default)
        if cfg_model is not None:
            self.cfg.update(cfg_model)

        self.backbone = build_backbone(model_name='mobilenetv2')
        self.fpn = build_fpn(model_name='fpn')
        self.ssh = build_ssh(model_name='ssh')
        self.head = build_head(model_name='sharehead')

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
