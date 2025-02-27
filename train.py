import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import numpy as np
import os
import pickle
import copy
from utils.pruning_utils import print_layer_info
from utils.pruning_utils import create_mask
from utils.pruning_utils import prune_and_reset
from models.MIRLModel import MIRLModel
from options.train_options import options as opt
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from utils.dataset_utils import TrainDataset
from utils.custom_callbacks import CustomCheckpoint
from torch.utils.data import DataLoader
import lightning.pytorch as pl


def main():
    # Ensure all modes are run if all-in-one is specified
    if "all_in_one" in opt.mode:
        opt.mode = ["denoise_15", "denoise_25", "denoise_50", "derain", "dehaze"]
    # Ensure all denoise modes are run if denoise is specified
    if "denoise" in opt.mode:
        opt.mode = [mode for mode in opt.mode if mode != "denoise"]
        opt.mode.extend(["denoise_15", "denoise_25", "denoise_50"])

    trainset = TrainDataset(opt)
    trainloader = DataLoader(
        trainset, 
        batch_size=opt.batch_size, 
        pin_memory=True, 
        shuffle=True, 
        drop_last=True, 
        num_workers=opt.num_workers
    )

    if opt.wdb_log is not None:
        logger = WandbLogger(project=opt.wdb_log, name="MIR-L-Train")
    else:
        logger = TensorBoardLogger(save_dir="logs/")

    # Initialize model
    model = MIRLModel(opt)

    if opt.initial_pruning_step > 0:
        # If starting from a later step, load the model checkpoint specified

        # ================================================= Note ========================================================
        # This expects that initial_state_dict.pth file exists in ckpt_dir
        # restore_ckpt_name is the name of the checkpoint to start training from,
        # if training is not starting from step 0 (no pruning) (for example step 1's - step 1 checkpoint)
        # ===============================================================================================================

        model_ckpt = torch.load(os.path.join(opt.ckpt_dir,opt.restore_ckpt_name), weights_only=True)
        model.load_state_dict(model_ckpt["state_dict"])
    
    else:
        # If starting from step 0,
        # store the initial state dict for restoring the surviving weights
        initial_state_dict = copy.deepcopy(model.state_dict())
        torch.save(initial_state_dict, (os.path.join(opt.ckpt_dir, "initial_state_dict.pth")))

    weight_mask = create_mask(model)

    initial_state_dict = None

    for step in range(opt.initial_pruning_step, opt.pruning_steps + 1):
        # Initial step (0) just trains the model, pruning begins from the first (1) step.
        # If step is > 0, prune the model and reset the surviving weights to the initial values
        if step != 0:
            # If the initial state to reset surviving weights to has not been loaded yet, load it
            if initial_state_dict == None:
                initial_state_dict = torch.load((os.path.join(opt.ckpt_dir, "initial_state_dict.pth")), weights_only=True)

            # Move model to GPU in case it is not, to ensure mask will also be on GPU
            model.to(torch.device("cuda:0"))

            # Prune the model and reset surviving weights to initial state
            prune_and_reset(model, weight_mask, opt.pruning_percent, opt.pruning_type, initial_state_dict)

            # Update the model's mask
            model.mask = weight_mask

        print(f"\n============= Pruning Step: {step} / {opt.pruning_steps} =============\n")

        # If verbose output is selected, print the table of nonzeros in each layer
        if opt.verbose:
            print_layer_info(model)

        # Print parameter info, compression rate and %
        total_params = 0
        zero_params = 0
        non_zero_params = 0

        for param in model.parameters():
            if param.requires_grad:
                total_params += param.numel()
                zero_params += torch.sum(param == 0).item()
                non_zero_params += torch.sum(param != 0).item()

        print("\n=======================================================")
        print(f"Total trainable parameters: {total_params}")
        print(f"Zero parameters: {zero_params}")
        print(f"Non-zero parameters: {non_zero_params}")
        print(f"Compression rate : {total_params/non_zero_params:.2f}x ({100 * (total_params-non_zero_params) / total_params:.2f}% pruned)")
        print("=======================================================\n")

        # Set up checkpoint callback for this step
        # Step 0: Initial training without pruning
        # Step 1, 2, 3 ...: Pruning steps (iterations)
        checkpoint_callback = CustomCheckpoint(
            every_n_epochs=opt.save_every_n_epochs,
            dirpath=opt.ckpt_dir,
            step=step
        )   

        # Set up trainer for this step
        trainer = pl.Trainer(
            max_epochs=opt.epochs, 
            accelerator="gpu", 
            devices=opt.num_gpus, 
            strategy="ddp_find_unused_parameters_true", 
            logger=logger, 
            callbacks=[checkpoint_callback]
        )

        # Train model
        trainer.fit(model=model, train_dataloaders=trainloader)

if __name__ == '__main__':
    main()