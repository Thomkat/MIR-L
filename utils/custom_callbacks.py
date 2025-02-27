import os
import lightning.pytorch as pl

# Saves checkpoints at regular intervals (every n epochs) with a stepped naming manner.
class CustomCheckpoint(pl.Callback):
    def __init__(self, every_n_epochs, dirpath, step):
        super().__init__()
        self._every_n_epochs = every_n_epochs
        self.dirpath = dirpath
        self.step = step
        os.makedirs(self.dirpath, exist_ok=True)  # Ensure save directory exists

    def on_train_epoch_end(self, trainer, pl_module):
        current_epoch = trainer.current_epoch
        # Save checkpoint every n epochs
        if self._every_n_epochs >= 1 and (current_epoch + 1) % self._every_n_epochs == 0:
            filename = f'model-epoch-{(current_epoch):02d}-step{self.step}.ckpt'
            filepath = os.path.join(self.dirpath, filename)
            trainer.save_checkpoint(filepath)