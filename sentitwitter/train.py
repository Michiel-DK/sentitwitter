import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from sentitwitter.data import DataModule
from sentitwitter.model import ClassModel

from pytorch_lightning.loggers import WandbLogger

import wandb
import pandas as pd

class SamplesVisualisationLogger(pl.Callback):
    def __init__(self, datamodule):
        super().__init__()

        self.datamodule = datamodule

    def on_validation_end(self, trainer, pl_module):
        val_batch = next(iter(self.datamodule.val_dataloader()))
        sentences = val_batch["text"]

        outputs = pl_module(val_batch["input_ids"].to('mps'), val_batch["attention_mask"].to('mps'), val_batch['labels'].to('mps'))
        preds = torch.argmax(outputs.logits, 1)
        labels = val_batch["labels"]

        df = pd.DataFrame(
            {"Sentence": sentences, "Label": labels.cpu().numpy(), "Predicted": preds.cpu().numpy()}
        )

        wrong_df = df[df["Label"] != df["Predicted"]]
        trainer.logger.experiment.log(
            {
                "examples": wandb.Table(dataframe=wrong_df, allow_mixed_types=True),
                "global_step": trainer.global_step,
            }
        )

def main():
    class_data = DataModule()
    class_model = ClassModel()

    checkpoint_callback = ModelCheckpoint(
        dirpath="./models", filename="best-checkpoint.ckpt",monitor="valid/loss", mode="min"
    )
    early_stopping_callback = EarlyStopping(
        monitor="valid/loss", patience=3, verbose=True, mode="min"
    )
    
    wandb_logger = WandbLogger(project='sentitwitter')

    trainer = pl.Trainer(
        #accelerator='cpu',#("gpu" if torch.cuda.is_available() else 'cpu'),
        accelerator = 'mps',
        devices = 1,
        max_epochs=1,
        fast_dev_run=False,
        #logger=pl.loggers.TensorBoardLogger("logs/", name="class", version=1),
        logger = wandb_logger,
        callbacks=[checkpoint_callback, early_stopping_callback, SamplesVisualisationLogger(class_data)],
        log_every_n_steps=10,
        deterministic=True,
        num_sanity_val_steps=1
    )
    trainer.fit(class_model, class_data)


if __name__ == "__main__":
    try:
        main()
        
    except Exception as e:
            import ipdb, traceback, sys
            extype, value, tb = sys.exc_info()
            traceback.print_exc()
            ipdb.post_mortem(tb)