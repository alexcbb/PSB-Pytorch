import hydra

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from models.psb import PSBModule
from datasets.cater import CaterDataModule

@hydra.main(config_path="configs", config_name="config")
def main(cfg):

    # Logger & Callbacks
    wandb_logger = WandbLogger(project="PSB", offline=cfg.offline, name="PSB_CATER")
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        monitor="train_loss",
        mode="min",
        dirpath=f"./checkpoints/psb",
        filename="{epoch:02d}-{train_loss:.2f}",
        save_weights_only=True, # only save the weights of the model
        every_n_epochs=2
    )
    # Model & Data
    model = PSBModule(cfg)
    datamodule = CaterDataModule(cfg)

    trainer_args = {
        'accelerator': 'gpu', 
        'devices': 1, 
        'max_steps': cfg.num_train_steps,
        'log_every_n_steps': cfg.log_every_steps,
        'logger': wandb_logger,
        'precision': '16-mixed',
        'benchmark': True,
    }
    trainer = L.Trainer(callbacks=[lr_monitor, checkpoint_callback], **trainer_args)
    trainer.fit(model, datamodule=datamodule)