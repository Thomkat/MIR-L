import argparse

parser = argparse.ArgumentParser()

# Input Parameters / Settings
parser.add_argument('--epochs', type=int, default=120, 
                    help='Total number of epochs for training the model.')
parser.add_argument('--warmup_epochs', type=int, default=15, 
                    help='Number of initial epochs for warmup during training. Used to gradually increase the learning rate.')
parser.add_argument('--batch_size', type=int, default=8, 
                    help="Batch size for each GPU during training.")
parser.add_argument('--lr', type=float, default=2e-4, 
                    help='Peak learning rate used by the linear warmup cosine annealing scheduler.')
parser.add_argument('--mode', nargs='+', default=['all_in_one'], 
                    help='Specifies the type of degradation scenarios the model will train for. Options: '
                         'denoise_15 (Sigma=15), denoise_25 (Sigma=25), denoise_50 (Sigma=50), denoise (train for all noise levels),'
                         'derain, dehaze, all_in_one (train for all tasks).')
parser.add_argument('--patch_size', type=int, default=64, 
                    help='Size of the image patches to be used as input during training. (Height = Width = patch_size)')
parser.add_argument('--num_workers', type=int, default=10, 
                    help='Number of worker threads for data loading.')
parser.add_argument("--num_gpus", type=int, default=1, 
                    help="Number of GPUs to utilize for training.")
parser.add_argument("--wdb_log", type=str, default=None, 
                    help='Project name for logging training progress to Weights & Biases (Wandb). '
                    'If set to None, Wandb logging is disabled and logs will be saved locally.')
parser.add_argument("--ckpt_dir", type=str, default="./checkpoints", 
                    help="Directory to save model checkpoints during training.")
parser.add_argument("--save_every_n_epochs", type=int, default=5, 
                    help="Interval (in epochs) to save model checkpoints.")
parser.add_argument("--pruning_steps", type=int, default=15, 
                    help="Number of incremental pruning rounds (steps) to apply during training.")
parser.add_argument("--initial_pruning_step", type=int, default=0, 
                    help="Pruning step to resume from when loading a pre-pruned model.")
parser.add_argument("--restore_ckpt_name", type=str, default=None, 
                    help="Name of the checkpoint to restore if initial_pruning_step is defined.")                    
parser.add_argument("--pruning_percent", type=int, default=20, 
                    help="Percentage of weights to prune in each pruning step.")
parser.add_argument("--pruning_type", type=str, default="global", 
                    help="Specifies whether to apply global or layer-wise pruning.")
parser.add_argument("--verbose", type=int, default=0, 
                    help="Level of verbosity for logging. Setting this to 1 prints details about the layers after each pruning step.")

# Dataset Options
parser.add_argument('--denoise_train_dir', type=str, default='./datasets/train/denoise', 
                    help='Path to the directory containing training datasets for denoising tasks.')
parser.add_argument('--derain_train_dir', type=str, default='./datasets/train/derain', 
                    help='Path to the directory containing training datasets for deraining tasks.')
parser.add_argument('--dehaze_train_dir', type=str, default='./datasets/train/dehaze', 
                    help='Path to the directory containing training datasets for dehazing tasks.')
parser.add_argument("--denoise_train_datasets", type=str, nargs='+', default=["WED", "BSD400"], 
                    help="List of dataset names to use for denoising training.")
parser.add_argument("--derain_train_datasets", type=str, nargs='+', default=["Rain100L"], 
                    help="List of dataset names to use for deraining training.")
parser.add_argument("--dehaze_train_datasets", type=str, nargs='+', default=["OTS"], 
                    help="List of dataset names to use for dehazing training.")
parser.add_argument("--denoise_data_augmentation", type=int, default=3, 
                    help="Apply data augmentation to balance datasets, in case some are smaller (1 = No Augmentation).")
parser.add_argument("--derain_data_augmentation", type=int, default=120, 
                    help="Apply data augmentation to balance datasets, in case some are smaller (1 = No Augmentation).")
parser.add_argument("--dehaze_data_augmentation", type=int, default=1, 
                    help="Apply data augmentation to balance datasets, in case some are smaller (1 = No Augmentation).")

options = parser.parse_args()
