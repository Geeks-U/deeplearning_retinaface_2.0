import torch
import torch.nn as nn
import copy

# 注册表
HEAD_REGISTRY = {}
# 装饰器
def register_head(name: str):
    def decorator(cls):
        HEAD_REGISTRY[name] = cls
        return cls
    return decorator


class ConvHead(nn.Module):
    """Shared head for bbox, cls, or landmark"""
    def __init__(self, in_channels, out_channels, num_anchors=2, num_layers=4):
        super().__init__()
        layers = []
        for _ in range(num_layers - 1):
            layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
        self.conv = nn.Sequential(*layers)
        self.out_channels = out_channels
        self.num_anchors = num_anchors

    def forward(self, x):
        out = self.conv(x)
        out = out.permute(0,2,3,1).contiguous()
        return out.view(out.shape[0], -1, int(self.out_channels/self.num_anchors))

# 对于每个 anchor：
# bbox: 4坐标（x, y, w, h）
# cls: 2分类（人脸/非人脸）
# ldm: 5个关键点 * 2坐标 (x, y)
cfg_default = {
    'in_channels': 256,
    'out_names': ['bbox' , 'cls', 'ldm'],
    'out_channels': [4, 2, 10],
    'num_anchor': 2
}

@register_head('sharehead')
class ShareHead(nn.Module):
    def __init__(self, cfg_head=None):
        super().__init__()
        self.cfg = copy.deepcopy(cfg_default)
        if cfg_head is not None:
            self.cfg.update(cfg_head)

        self.bbox_head = ConvHead(self.cfg['in_channels'], self.cfg['num_anchor'] * self.cfg['out_channels'][0], self.cfg['num_anchor'])
        self.cls_head = ConvHead(self.cfg['in_channels'], self.cfg['num_anchor'] * self.cfg['out_channels'][1], self.cfg['num_anchor'])
        self.ldm_head = ConvHead(self.cfg['in_channels'], self.cfg['num_anchor'] * self.cfg['out_channels'][2], self.cfg['num_anchor'])

    def forward(self, x):
        # features: List[Tensor], 每层 FPN 特征图
        bbox_outputs = []
        cls_outputs = []
        ldm_outputs = []

        for key in ['high', 'mid', 'low']:
            bbox_outputs.append(self.bbox_head(x[key]))
            cls_outputs.append(self.cls_head(x[key]))
            ldm_outputs.append(self.ldm_head(x[key]))

        # 使用 torch.cat 对每个输出进行拼接，这里假设在 dim=1（即特征维度）上拼接
        bbox_outputs = torch.cat(bbox_outputs, dim=1)
        cls_outputs = torch.cat(cls_outputs, dim=1)
        ldm_outputs = torch.cat(ldm_outputs, dim=1)

        return {
            self.cfg['out_names'][0]: bbox_outputs,
            self.cfg['out_names'][1]: cls_outputs,
            self.cfg['out_names'][2]: ldm_outputs
            }

def build_head(model_name: str, cfg_head=None) -> nn.Module:
    if model_name not in HEAD_REGISTRY:
        raise ValueError(f"Unknown model name: {model_name}")
    return HEAD_REGISTRY[model_name](cfg_head=cfg_head)

from src.nets.backbone import build_backbone
from src.nets.fpn import build_fpn
from src.nets.ssh import build_ssh

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


    head = build_head(model_name='sharehead')
    head.eval()

    with torch.no_grad():
        outputs = head(outputs)
    print("head模型配置:", head.cfg)

    print("各输出特征图形状：")
    for i, name in enumerate(['bbox', 'cls', 'ldm']):
        feat = outputs[name]
        print(f"{name}: {feat.shape}")
