import torch
import torchvision.models as models
import copy
from collections import OrderedDict

cfg_default = {
    'pretrained': True,
    'frozen': False,
    'out_layers': [4, 7, 14],  # 需根据实际模型结构调整
    'out_step': None            # 支持手动指定或自动计算
}

class MobileNetV2(torch.nn.Module):
    """MobileNetV2 backbone with configurable output layers and stride steps.

    Args:
        cfg_mobileNetV2 (dict): Configuration overrides.
            Valid keys: 'pretrained', 'frozen', 'out_layers', 'out_step'.
    """
    def __init__(self, cfg_mobileNetV2=None):
        super().__init__()
        self.cfg = copy.deepcopy(cfg_default)
        if cfg_mobileNetV2 is not None:
            self.cfg.update(cfg_mobileNetV2)

        # Load model
        weights = models.MobileNet_V2_Weights.DEFAULT if self.cfg['pretrained'] else None
        self.model = models.mobilenet_v2(weights=weights)

        # Freeze parameters
        if self.cfg['frozen']:
            for param in self.model.parameters():
                param.requires_grad = False

        # Validate out_layers indices
        max_layer_idx = len(self.model.features) - 1
        invalid_indices = [idx for idx in self.cfg['out_layers'] if idx > max_layer_idx]
        if invalid_indices:
            raise ValueError(f"Invalid layer indices: {invalid_indices}")

        # 计算或校验 out_step
        if self.cfg['out_step'] is not None:
            if len(self.cfg['out_step']) != len(self.cfg['out_layers']):
                raise ValueError("out_step 必须与 out_layers 长度一致")
        else:
            self.cfg['out_step'] = self._compute_out_steps()

    def _compute_out_steps(self):
        """自动计算输出层的实际步长"""
        dummy_input = torch.randn(1, 3, 224, 224)
        strides = []
        current_stride = 1
        with torch.no_grad():
            for idx, layer in enumerate(self.model.features):
                # 记录前向传播前的尺寸
                input_size = dummy_input.shape[-1]
                dummy_input = layer(dummy_input)
                output_size = dummy_input.shape[-1]
                # 计算当前层的实际步长（输入尺寸 / 输出尺寸）
                layer_stride = input_size // output_size
                current_stride *= layer_stride
                if idx in self.cfg['out_layers']:
                    strides.append(current_stride)
        return strides

    @property
    def out_steps(self):
        """返回输出特征图的步长列表（相对于输入图像）"""
        return self.cfg['out_step']

    def forward(self, x):
        output = OrderedDict()
        for idx, layer in enumerate(self.model.features):
            x = layer(x)
            if idx in self.cfg['out_layers']:
                output[f"layer_{idx}"] = x
        return output

    @staticmethod
    def test():
        """测试模型初始化、前向传播及步长计算"""
        print(f'Is CUDA available? : {torch.cuda.is_available()}')

        model = MobileNetV2()
        model.eval()

        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            outputs = model(x)

        print("Is pretrained? :", model.cfg['pretrained'])
        print("Is frozen? :", model.cfg['frozen'])
        print("Output feature shapes from out_layers:")
        for i, (idx, feat) in enumerate(outputs.items()):
            print(f"Layer {idx}: {feat.shape} (stride={model.out_steps[i]})")

if __name__ == "__main__":
    MobileNetV2.test()
