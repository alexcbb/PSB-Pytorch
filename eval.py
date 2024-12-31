import hydra

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from models.psb import PSBModule
from datasets.dataset import VideoFolderDataModule

@hydra.main(config_path="configs", config_name="config")
def main(cfg):

    # Logger & Callbacks
    wandb_logger = WandbLogger(project="SlotFormer", offline=cfg.offline, name=f"PSB_{cfg.dataset_name}")

    assert cfg.checkpoint_path is not None, "Please provide a checkpoint path"

    # Model & Data
    model = PSBModule(cfg)
    model = model.load_from_checkpoint(cfg.checkpoint_path)
    model.eval()
    datamodule = VideoFolderDataModule(cfg)

    trainer_args = {
        'accelerator': 'gpu', 
        'devices': 1, 
        'log_every_n_steps': cfg.log_every_steps,
        'logger': wandb_logger,
    }
    trainer = L.Trainer(
        **trainer_args
        )
    trainer.validate(model, datamodule=datamodule)

if __name__ == "__main__":
    main()