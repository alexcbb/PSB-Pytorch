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
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        monitor="train_loss",
        mode="min",
        dirpath=f"./checkpoints/psb",
        filename=f"PSB_{cfg.dataset_name}_"+"{epoch:02d}-{train_loss:.2f}",
        save_weights_only=True, # only save the weights of the model
        every_n_train_steps=cfg.checkpoint_every_n_steps,
    )
    # Model & Data
    model = PSBModule(cfg)
    datamodule = VideoFolderDataModule(cfg)

    # TODO : update training args
    trainer_args = {
        'accelerator': 'gpu', 
        'devices': 1, 
        'max_steps': cfg.num_train_steps,
        'log_every_n_steps': cfg.log_every_steps,
        'logger': wandb_logger,
        'val_check_interval': cfg.val_check_interval,
        'max_epochs': -1,  # We control training duration using `max_steps`
        'check_val_every_n_epoch':None,  # We do not use epochs for training
    }
    trainer = L.Trainer(
        callbacks=[lr_monitor, checkpoint_callback], 
        **trainer_args
        )
    trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    main()