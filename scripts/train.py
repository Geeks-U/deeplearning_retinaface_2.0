from src.train.trainer import Trainer

if __name__ == '__main__':
    cfg_trainer = {
        'num_epochs': 30,
        'batch_size': 32,
    }

    trainer = Trainer(cfg_trainer=cfg_trainer)
    trainer.train()
