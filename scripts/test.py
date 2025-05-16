from src.test.predict import run_single_image_predict

cfg_test = {
    'cfg_net': {
        'mode': 'eval',
        'cfg_backbone': {
            'pretrained': True,
            'out_stage': [4, 7, 14],
            'out_channels': [32, 64, 160],
            'steps': [8, 16, 32],
            'unfreeze': False,
            'unfreeze_stage': [4, 7, 14]
        },
        'cfg_fpn': {
            'pretrained': False,
            'in_channels': [32, 64, 160],
            'out_channels': [64, 64, 64],
            'steps': [1, 1, 1]
        },

        'cfg_ssh': {
            'pretrained': False,
            'in_channels': [64, 64, 64],
            'out_channels': [256, 256, 256],
            'steps': [1, 1, 1]
        },

        'cfg_head': {
            'pretrained': False,
            'num_anchor': 2,
            'in_channels': [256],
            'out_channels': [4, 2, 10],
            'steps': []
        },
    },
# self.num_fpn_feature_layers = 3
# self.backbone_fpn_strides = [8, 16, 32]
# self.num_anchor_per_pixel = 2
# self.anchor_ratios_per_level = [[8, 16], [32, 64], [128, 256]]
# self.input_image_size = [320, 320]
# self.clip = False
    'cfg_anchor': {
        'input_image_size': [1280, 1280],
        'num_fpn_feature_layers': 3,
        'backbone_fpn_strides': [8, 16, 32],
        'num_anchor_per_pixel': 2,
        'anchor_ratios_per_level': [[8, 16], [32, 64], [128, 256]],
        'clip': False,
        'variance': [0.1, 0.2]
    },
    # checkpoints/1746204358.3167648_best.pth
    'cfg_data': {
        'model_path': r'D:\Code\DL\Pytorch\retinaface\weights\model_last_20250516_165547.pth',
        'img_path': r'D:\Code\DL\Pytorch\retinaface\src\images\0_Parade_marchingband_1_122.jpg',
        'image_size': [1280, 1280],
    },
    'cfg_hyperparameter': {
        'CUDA': True,
        'letterbox_image': True,
        'confidence': 0.5
    },
}

# 入口
if __name__ == "__main__":
    # 单个图片测试
    run_single_image_predict(cfg_test=cfg_test)
