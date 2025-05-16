from src.training.trainer import train
import time

cfg_train = {
    'cfg_net': {
        'mode': 'train',
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
    'cfg_anchor': {
        'input_image_size': [320, 320],
        'num_fpn_feature_layers': 3,
        'backbone_fpn_strides': [8, 16, 32],
        'num_anchor_per_pixel': 2,
        'anchor_ratios_per_level': [[8, 16], [32, 64], [128, 256]],
        'clip': False,
        # xy, wh的缩放倍数
        'variance': [0.1, 0.2]
    },
    'cfg_data': {
        'model_save_dir': r'D:\Code\Python\deeplearning\retinaface\checkpoints',
        'model_save_filename_prefix': str(time.time()),
        'images_dir': r'D:\Code\Python\deeplearning\data\widerface\train_val\images',
        'images_label_path': r'D:\Code\Python\deeplearning\data\widerface\train_val\label.txt',
        'input_image_size': [320, 320],
        'data_split': [0.8, 0.1, 0.1]
    },
    'cfg_hyperparameter': {
        'CUDA': True,
        'lr': 1e-2,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'scheduler_step_size': 10,
        'scheduler_gamma': 0.1,
        'epochs': 15,
        'batch_size': 32,
        'num_workers': 4,
        'drop_last': True


    },
    'cfg_loss': {
        'overlap_thresh': 0.35,
        'neg_pos': 7,
    }
}

# 入口
if __name__ == "__main__":
    start_time = time.time()  # 记录开始时间
    train(cfg_train=cfg_train)
    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time
    print(f"训练函数运行时间: {elapsed_time} 秒")