from src.train.trainer import Trainer

if __name__ == '__main__':
    # input_image_size在dataset以及anchor生成都需要使用，所以进行统一处理
    cfg_trainer = {
        'num_epochs': 30,
        'batch_size': 32,
        'input_image_size': [320, 320],
    }

    trainer = Trainer(cfg_trainer=cfg_trainer)
    trainer.train()
