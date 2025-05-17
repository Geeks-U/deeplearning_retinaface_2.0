# backbone.py
import copy
import torch
import torch.nn as nn
import torchvision.models as models


# 注册表
BACKBONE_REGISTRY = {}
# 装饰器
def register_backbone(name: str):
    def decorator(cls):
        BACKBONE_REGISTRY[name] = cls
        return cls
    return decorator


# 默认配置
cfg_mobilenetv2_default = {
    'pretrained': True,
    'frozen': False,
    'out_names': ['high', 'mid', 'low'],
    'out_layers': [4, 7, 14],
    'out_steps': [8, 16, 32],
    'out_channels': [32, 64, 160]
}

@register_backbone('mobilenetv2')
class MobileNetV2(nn.Module):
    def __init__(self, cfg_backbone=None):
        super().__init__()
        self.cfg = copy.deepcopy(cfg_mobilenetv2_default)
        if cfg_backbone is not None:
            self.cfg.update(cfg_backbone)

        weights = models.MobileNet_V2_Weights.DEFAULT if self.cfg['pretrained'] else None
        self.model = models.mobilenet_v2(weights=weights)

        if self.cfg['frozen']:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x):
        outputs = {}
        out_indices = self.cfg['out_layers']
        out_names = self.cfg['out_names']
        name_map = dict(zip(out_indices, out_names))
        max_layer = max(out_indices)
        for idx, layer in enumerate(self.model.features):
            x = layer(x)
            if idx in name_map:
                outputs[name_map[idx]] = x
            if idx >= max_layer:
                break
        return {name: outputs[name] for name in out_names}

# 工厂函数
def build_backbone(model_name: str, cfg_backbone=None) -> nn.Module:
    if model_name not in BACKBONE_REGISTRY:
        raise ValueError(f"Unknown model name: {model_name}")
    return BACKBONE_REGISTRY[model_name](cfg_backbone=cfg_backbone)

# 测试用例
if __name__ == "__main__":
    model = build_backbone('mobilenetv2', cfg_backbone={'pretrained': False})
    print("模型配置:", model.cfg)
    model.eval()

    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        outputs = model(x)

    print("各输出特征图形状：")
    for i, name in enumerate(['high', 'mid', 'low']):
        feat = outputs[name]
        print(f"{name}: {feat.shape} (stride={model.cfg['out_steps'][i]})")
