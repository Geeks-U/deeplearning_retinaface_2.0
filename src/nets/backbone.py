# 框架导入
import torch
# 模型导入
import torchvision.models as models
# 工具导入
import copy

cfg_default = {
    'pretrained': True,
    'frozen': False,
    'out_layers': [4, 7, 14],
    'out_channels': [32, 64, 160],
    'out_step': [8, 16, 32]
}

class MobileNetV2(torch.nn.Module):
    def __init__(self, cfg_mobileNetV2=None):
        # 父类初始化
        super(MobileNetV2, self).__init__()

		# 参数加载
        self.cfg = copy.deepcopy(cfg_default if cfg_mobileNetV2 is None else cfg_mobileNetV2)

        # 加载模型
        self.model = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.DEFAULT if self.cfg['pretrained'] else None
        )

        # 冻结参数
        if self.cfg['frozen']:
            for child in self.model.features:
                for param in child.parameters():
                    param.requires_grad = False

    def forward(self, x):
        output = []
        for i, child in enumerate(self.model.features):
            x = child(x)
            if i in self.cfg['out_layers']:
                output.append(x)
        return output

    @staticmethod
    def test():
        print(f'Is CUDA available? : {torch.cuda.is_available()}')

        model = MobileNetV2()
        model.eval()

        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            outputs = model(x)

        print("Is pretrained? :", model.cfg['pretrained'])
        print("Is frozen? :", model.cfg['frozen'])
        print("Output feature shapes from out_layers:")
        for i, feat in enumerate(outputs):
            print(f"Layer index {model.cfg['out_layers'][i]}: {feat.shape}")


if __name__ == "__main__":
    MobileNetV2.test()
