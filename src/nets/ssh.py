import torch
import torch.nn as nn
import copy

class SSHBasic(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        assert out_channels % 4 == 0, "out_channels must be divisible by 4"

        # 主分支（Identity Branch）
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=1, stride=1, padding=0)

        # 上下文分支1（Context Branch 1）
        self.conv2 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=3, stride=1, padding=2, dilation=2)

        # 上下文分支2（Context Branch 2）
        self.conv4 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=3, stride=1, padding=3, dilation=3)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 主分支
        identity = self.conv1(x)

        # 上下文分支1
        conv2 = self.conv2(x)
        context1 = self.conv3(conv2)

        # 上下文分支2
        conv4 = self.conv4(x)
        context2 = self.conv5(conv4)

        # 拼接所有分支
        outputs = torch.cat([identity, context1, context2], dim=1)
        outputs = self.relu(outputs)

        return outputs

cfg_default = {
    'in_channels': [64, 64, 64],
    'out_channels': [256, 256, 256],
}

class SSH(nn.Module):
    def __init__(self, cfg_ssh=None):
        super().__init__()
        self.cfg = copy.deepcopy(cfg_default)
        if cfg_ssh is not None:
            self.cfg.update(cfg_ssh)
        self.stage_high = SSHBasic(in_channels=self.cfg['in_channels'][0], out_channels=self.cfg['out_channels'][0])
        self.stage_mid = SSHBasic(in_channels=self.cfg['in_channels'][1], out_channels=self.cfg['out_channels'][1])
        self.stage_low = SSHBasic(in_channels=self.cfg['in_channels'][2], out_channels=self.cfg['out_channels'][2])

    def forward(self, x):
        output_stage_low = self.stage_low(x['low'])
        output_stage_mid = self.stage_mid(x['mid'])
        output_stage_high = self.stage_high(x['high'])
        return {
            'high': output_stage_high,
            'mid': output_stage_mid,
            'low': output_stage_low
        }

from src.nets.backbone import MobileNetV2
from src.nets.fpn import FPN

if __name__ == '__main__':
    mn = MobileNetV2()
    mn.eval()

    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        outputs = mn(x)
    print("mn模型配置:", mn.cfg)

    print("各输出特征图形状：")
    for i, name in enumerate(['high', 'mid', 'low']):
        feat = outputs[name]
        print(f"{name}: {feat.shape} (stride={mn.out_steps[i]})")

    fpn = FPN()
    fpn.eval()

    with torch.no_grad():
        outputs = fpn(outputs)
    print("fpn模型配置:", fpn.cfg)

    print("各输出特征图形状：")
    for i, name in enumerate(['high', 'mid', 'low']):
        feat = outputs[name]
        print(f"{name}: {feat.shape} (stride={fpn.out_steps[i]})")

    ssh = SSH()
    ssh.eval()

    with torch.no_grad():
        outputs = ssh(outputs)
    print("ssh模型配置:", ssh.cfg)

    print("各输出特征图形状：")
    for i, name in enumerate(['high', 'mid', 'low']):
        feat = outputs[name]
        print(f"{name}: {feat.shape}")
