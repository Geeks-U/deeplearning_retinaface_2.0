import torch
import torch.nn as nn
import copy

# 注册表
SSH_REGISTRY = {}
# 装饰器
def register_ssh(name: str):
    def decorator(cls):
        SSH_REGISTRY[name] = cls
        return cls
    return decorator


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

cfg_ssh_default = {
    'in_channels': [64, 64, 64],
    'out_names': ['high', 'mid', 'low'],
    'out_channels': [256, 256, 256],
}

@register_ssh('ssh')
class SSH(nn.Module):
    def __init__(self, cfg_ssh=None):
        super().__init__()
        self.cfg = copy.deepcopy(cfg_ssh_default)
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
            self.cfg['out_names'][0]: output_stage_high,
            self.cfg['out_names'][1]: output_stage_mid,
            self.cfg['out_names'][2]: output_stage_low
        }

# 工厂函数
def build_ssh(model_name: str, cfg_ssh=None) -> nn.Module:
    if model_name not in SSH_REGISTRY:
        raise ValueError(f"Unknown model name: {model_name}")
    return SSH_REGISTRY[model_name](cfg_ssh=cfg_ssh)

from src.nets.backbone import build_backbone
from src.nets.fpn import build_fpn

if __name__ == '__main__':
    x = torch.randn(1, 3, 640, 640)

    mn = build_backbone(model_name='mobilenetv2', cfg_backbone={'pretrained': True})
    mn.eval()

    with torch.no_grad():
        outputs = mn(x)
    print("mn模型配置:", mn.cfg)

    print("各输出特征图形状：")
    for i, name in enumerate(['high', 'mid', 'low']):
        feat = outputs[name]
        print(f"{name}: {feat.shape} (stride={mn.cfg['out_steps'][i]})")

    fpn = build_fpn(model_name='fpn', cfg_fpn={'out_steps': [1, 1, 1]})
    fpn.eval()

    with torch.no_grad():
        outputs = fpn(outputs)
    print("fpn模型配置:", fpn.cfg)

    print("各输出特征图形状：")
    for i, name in enumerate(['high', 'mid', 'low']):
        feat = outputs[name]
        print(f"{name}: {feat.shape} (stride={fpn.cfg['out_steps'][i]})")

    ssh = build_ssh(model_name='ssh')
    ssh.eval()

    with torch.no_grad():
        outputs = ssh(outputs)
    print("ssh模型配置:", ssh.cfg)

    print("各输出特征图形状：")
    for i, name in enumerate(['high', 'mid', 'low']):
        feat = outputs[name]
        print(f"{name}: {feat.shape}")
