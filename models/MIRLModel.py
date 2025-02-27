import lightning.pytorch as pl
from models.network import PromptIR
import torch.nn as nn
import torch.optim as optim
from utils.schedulers import LinearWarmupCosineAnnealingLR
import torch

class MIRLModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.net = PromptIR(decoder=True).to(torch.device("cuda:0"))
        self.loss_fn  = nn.L1Loss()
        self.args = args

        # We'll later set self.mask once pruning is done
        self.mask = None
    
    def forward(self,x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)

        loss = self.loss_fn(restored,clean_patch)
        # Logging to TensorBoard by default
        self.log("train_loss", loss)

        return loss
    
    def lr_scheduler_step(self,scheduler,metric):
        scheduler.step()
        lr = scheduler.get_last_lr()
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.args.lr)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer, warmup_epochs=self.args.warmup_epochs, max_epochs=self.args.epochs)

        return [optimizer],[scheduler]
    
    def on_after_backward(self):
        #Freeze pruned weights by zeroing their gradients.
        if self.mask is None:
            return  # No pruning applied yet

        weight_params = [p for n, p in self.named_parameters() if 'weight' in n]
        
        for param, m in zip(weight_params, self.mask):
            if param.grad is not None:
                param.grad.data.mul_(m)
    
    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)

        if self.mask is None:
            return  # No pruning applied yet
        
        # Mask weights
        weight_params = [p for n, p in self.named_parameters() if 'weight' in n]

        if self.mask is not None:
            with torch.no_grad():
                for param, m in zip(weight_params, self.mask):
                    param.data.mul_(m)