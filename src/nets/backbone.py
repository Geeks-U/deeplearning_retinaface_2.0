import torch
import torch.nn as nn
import torchvision.models as models
import copy

# 默认配置
cfg_default = {
    'pretrained': True,               # 是否使用预训练权重
    'frozen': False,                  # 是否冻结特征提取层
    'out_layers': [4, 7, 14],         # 指定哪些层的输出作为特征输出
    'out_step': None,                 # 每个输出特征的步长（stride）
    'out_channels': None              # 每个输出特征的通道数
}

class MobileNetV2(nn.Module):
    """MobileNetV2 主干网络，可配置输出层、逻辑名称、步长和通道数。"""

    def __init__(self, cfg_mobileNetV2=None):
        super().__init__()
        # 合并用户配置和默认配置
        self.cfg = copy.deepcopy(cfg_default)
        if cfg_mobileNetV2 is not None:
            self.cfg.update(cfg_mobileNetV2)

        # 加载预训练模型
        weights = models.MobileNet_V2_Weights.DEFAULT if self.cfg['pretrained'] else None
        self.model = models.mobilenet_v2(weights=weights)

        # 是否冻结模型参数
        if self.cfg['frozen']:
            for param in self.model.parameters():
                param.requires_grad = False

        # 计算输出步长
        self.cfg['out_step'] = self._compute_out_steps()

        # 计算每个输出层的通道数
        self.cfg['out_channels'] = self._get_out_channels()

    def _compute_out_steps(self):
        """自动计算每个输出层的步长（stride）"""
        dummy_input = torch.randn(1, 3, 224, 224)
        strides = []
        current_stride = 1
        with torch.no_grad():
            for idx, layer in enumerate(self.model.features):
                input_size = dummy_input.shape[-1]
                dummy_input = layer(dummy_input)
                output_size = dummy_input.shape[-1]
                layer_stride = input_size // output_size
                current_stride *= layer_stride
                if idx in self.cfg['out_layers']:
                    strides.append(current_stride)
        return strides

    def _get_out_channels(self):
        """获取每个输出层的通道数"""
        channels = []
        for idx in self.cfg['out_layers']:
            layer = self.model.features[idx]
            if isinstance(layer, torch.nn.Sequential):
                last_sub_layer = list(layer.children())[-1]
                channels.append(last_sub_layer.out_channels)
            else:
                channels.append(layer.out_channels)
        return channels

    @property
    def out_steps(self):
        return self.cfg['out_step']

    def forward(self, x):
        """前向传播，返回指定层的特征图，使用 high/mid/low 命名"""
        outputs = {}
        out_indices = self.cfg['out_layers']
        out_names = ['high', 'mid', 'low']
        name_map = dict(zip(out_indices, out_names))
        max_layer = max(out_indices)

        for idx, layer in enumerate(self.model.features):
            x = layer(x)
            if idx in name_map:
                outputs[name_map[idx]] = x
            if idx >= max_layer:
                break

        # 确保输出顺序为 high, mid, low
        return {name: outputs[name] for name in ['high', 'mid', 'low']}


# 测试入口
if __name__ == "__main__":
    model = MobileNetV2()
    print("模型配置:", model.cfg)
    model.eval()

    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        outputs = model(x)

    print("各输出特征图形状：")
    for i, name in enumerate(['high', 'mid', 'low']):
        feat = outputs[name]
        print(f"{name}: {feat.shape} (stride={model.out_steps[i]})")
