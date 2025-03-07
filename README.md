# MIR-L

This repository provides a PyTorch implementation of the MIR-L model proposed in the paper titled **Pruning Overparameterized Multi-Task Image Restoration Networks via Lottery Tickets**.

## Installation and Datasets
See [`INSTALL.md`](INSTALL.md) for details.

---

## Training

The training script is `train.py`.

### Important Training Arguments

Below are some commonly used arguments from `options/train_options.py`. You can either edit them directly or specify them on the command line:

- **`--epochs`**: Total number of epochs (default: 120).  
- **`--warmup_epochs`**: Warmup epochs for the learning rate scheduler (default: 15).  
- **`--batch_size`**: Batch size per GPU (default: 8).  
- **`--lr`**: Peak learning rate (default: 2e-4).  
- **`--mode`**: List of tasks to train for. Possible values:
  - `denoise_15, denoise_25, denoise_50`
  - `derain`
  - `dehaze`
  - `denoise` (automatically expands to 15, 25, 50)
  - `all_in_one` (runs all tasks: denoise at 15,25,50 + derain + dehaze)
- **`--num_gpus`**: Number of GPUs for PyTorch Lightning (default: 1).
- **`--wdb_log`**: If provided, sets the Wandb project name. If `None`, logs locally to `logs/`.
- **`--ckpt_dir`**: Directory to save checkpoints (`./checkpoints` by default).
- **`--pruning_steps`**: Number of pruning steps (default: 15).
- **`--pruning_percent`**: % of weights to prune each step (default: 20).
- **`--pruning_type`**: `global` or `layerwise`.
- **`--initial_pruning_step`**: If you want to **resume** from a certain pruning step, set this and also provide `--restore_ckpt_name`.

### Iterative Pruning Flow

- **Step 0**: Train the model without pruning for `--epochs` epochs. A checkpoint `initial_state_dict.pth` is saved to later restore weights from.  
- **Step 1 to N** (where `N = --pruning_steps`):
  1. Prune a given percentage of weights (i.e., 20%).  
  2. Reset surviving weights to their original values from step 0.  
  3. Re-train for another `--epochs` epochs. 
  4. Repeat i-iii

### Example Training Commands

- **1) Train all tasks (no pruning)**:
  ```bash
  python train.py \
    --mode all_in_one \
    --pruning_steps 0 \
    --epochs 120 \
    --batch_size 8 \
    --lr 2e-4 \
    --warmup_epochs 15 \
    --num_gpus 1
  ```

- **2) Train with iterative pruning** (15 steps, pruning 20% globally each time):
  ```bash
  python train.py \
    --mode all_in_one \
    --pruning_steps 15 \
    --pruning_percent 20 \
    --pruning_type global \
    --epochs 120 \
    --batch_size 8 \
    --lr 2e-4 \
    --warmup_epochs 15 \
    --num_gpus 1 \
    --ckpt_dir ./checkpoints
  ```

- **3) Resume from a specific pruning step**:
  Suppose you already finished up to step 3, and have a checkpoint named `model-epoch-119-step3.ckpt`. To resume from step 4:
  ```bash
  python train.py \
    --initial_pruning_step 4 \
    --restore_ckpt_name model-epoch-119-step3.ckpt \
    --pruning_steps 15
    ...
  ```

---

## Testing

Use `test.py` to run inference and evaluate model performance (PSNR and SSIM). It supports the same set of degradation modes and will output images and a tabulated final result.

### Important Testing Arguments

From `options/test_options.py`:

- **`--ckpt_dir`**: Directory containing the checkpoint to test.  
- **`--ckpt_name`**: Name of the checkpoint file (e.g. `model-epoch-119-step15.ckpt`).  
- **`--mode`**: Which tasks to test. Can be a list: `denoise_15`, `derain`, `all_in_one`, etc.  
- **`--output_dir`**: Where restored images will be saved. Default: `./output`.  
- **`--denoise_test_datasets`**: List of test sets for denoising (e.g. `BSD68`, `Urban100`).  
- **`--derain_test_datasets`**: List of test sets for deraining (e.g. `Rain100L`).  
- **`--dehaze_test_datasets`**: List of test sets for dehazing (e.g. `SOTS`).  

### Example Testing Commands

1. **Test on all tasks** (checkpoint from final pruning step):
   ```bash
   python test.py \
     --ckpt_dir ./checkpoints \
     --ckpt_name model-epoch-119-step15.ckpt \
     --mode all_in_one \
     --denoise_test_datasets BSD68 \
     --derain_test_datasets Rain100L \
     --dehaze_test_datasets SOTS \
     --output_dir ./output
   ```

2. **Test only deraining**:
   ```bash
   python test.py \
     --ckpt_dir ./checkpoints \
     --ckpt_name model-epoch-119-step15.ckpt \
     --mode derain \
     --derain_test_datasets Rain100L \
     --output_dir ./output
   ```

3. **Test only denoising with sigma=15,25,50**:
   ```bash
   python test.py \
     --ckpt_dir ./checkpoints \
     --ckpt_name model-epoch-119-step15.ckpt \
     --mode denoise \
     --denoise_test_datasets BSD68 Urban100 \
     --output_dir ./output
   ```

---

## Logging and Checkpoints

- **Weights & Biases (Wandb)**:  
  If you specify `--wdb_log <PROJECT_NAME>`, training logs are uploaded to your Wandb project.  
- **TensorBoard**:  
  If `--wdb_log` is `None`, logs are saved locally to `./logs/` (TensorBoard format).

- **Checkpoints**:  
  - Saved in `--ckpt_dir` (default: `./checkpoints`).  
  - Named in the format: `model-epoch-XX-stepY.ckpt`, where `Y` is the pruning step.  
  - The script also saves an `initial_state_dict.pth` after step 0, used to reset surviving weights after each pruning step.

---

## Acknowledgment:

This repository is based on the [PromptIR](https://github.com/va1shn9v/PromptIR) repository and some logic was inspired from the [Lottery-Ticket-Hypothesis-in-Pytorch](https://github.com/rahulvigneswaran/Lottery-Ticket-Hypothesis-in-Pytorch) repository.
