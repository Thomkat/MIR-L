import argparse

parser = argparse.ArgumentParser()

# Input Parameters / Settings
parser.add_argument('--output_dir', type=str, default="./output", 
                    help='Directory to save the output images generated during testing.')
parser.add_argument("--ckpt_dir", type=str, default="./pretrained_models", 
                    help="Directory where the model checkpoint files are stored for testing.")
parser.add_argument('--ckpt_name', type=str, default="all_in_one.ckpt", 
                    help='Name of the specific checkpoint file to load for testing.')
parser.add_argument('--mode', nargs='+', default=["all_in_one"], 
                    help='Specifies the type of degradation scenarios to test the model for. Options: '
                         'denoise_15 (Sigma=15), denoise_25 (Sigma=25), denoise_50 (Sigma=50), '
                         'derain, dehaze, all_in_one (train for all tasks).')
parser.add_argument('--num_workers', type=int, default=16, 
                    help='Number of worker threads for data loading.')

# Dataset Options
parser.add_argument('--denoise_test_dir', type=str, default='./datasets/test/denoise', 
                    help='Directory containing datasets for testing denoising tasks.')
parser.add_argument('--derain_test_dir', type=str, default='./datasets/test/derain', 
                    help='Directory containing datasets for testing deraining tasks.')
parser.add_argument('--dehaze_test_dir', type=str, default='./datasets/test/dehaze', 
                    help='Directory containing datasets for testing dehazing tasks.')
parser.add_argument("--denoise_test_datasets", type=str, nargs='+', default=["BSD68", "Urban100"], 
                    help="List of dataset names to use for testing denoising tasks.")
parser.add_argument("--derain_test_datasets", type=str, nargs='+', default=["Rain100L"], 
                    help="List of dataset names to use for testing deraining tasks.")
parser.add_argument("--dehaze_test_datasets", type=str, nargs='+', default=["SOTS"], 
                    help="List of dataset names to use for testing dehazing tasks.")

options = parser.parse_args()
