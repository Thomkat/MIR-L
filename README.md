# MIR-L

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2510.14463)
[![model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/Tomk187/MIR-L)

This repository provides a PyTorch implementation of the MIR-L model proposed in the paper titled **Pruning Overparameterized Multi-Task Networks for Degraded Web Image Restoration**.

> **Abstract:** Image quality is a critical factor in delivering visually appealing content on web platforms. However, images often suffer from degradation due to lossy operations applied by online social networks (OSNs), negatively affecting user experience. Image restoration is the process of recovering a clean high-quality image from a given degraded input. Recently, multi-task (all-in-one) image restoration models have gained significant attention, due to their ability to simultaneously handle different types of image degradations. However, these models often come with an excessively high number of trainable parameters, making them computationally inefficient. In this paper, we propose a strategy for compressing multi-task image restoration models. We aim to discover highly sparse subnetworks within overparameterized deep models that can match or even surpass the performance of their dense counterparts. The proposed model, namely MIR-L, utilizes an iterative pruning strategy that removes low-magnitude weights across multiple rounds, while resetting the remaining weights to their original initialization. This iterative process is important for the multi-task image restoration model’s optimization, effectively uncovering “winning tickets” that maintain or exceed state-of-the-art performance at high sparsity levels. Experimental evaluation on benchmark datasets for the deraining, dehazing, and denoising tasks shows that MIR-L retains only 10% of the trainable parameters while maintaining high image restoration performance.

---

## News / Updates
- *[Oct 2025]* Pre-trained models are available on Google Drive and 🤗 Hugging Face.
- *[Oct 2025]* Datasets are available.
- *[Sep 2025]* Paper accepted as a long paper at **WI-IAT 2025**.

---

## Installation and Datasets

Train/test datasets are available to download [**here**](https://drive.google.com/file/d/1NsDbszHF3OIJk_jMnXg2LuOqE8sXQvU8/view?usp=sharing).

See [`INSTALL.md`](INSTALL.md) for details.

---

## Pre-trained Models

We provide the following pre-trained MIR-L models for different restoration tasks:

| Task | Google Drive Link |
|------|-------------|
| **Denoising** | [MIR-L Denoise](https://drive.google.com/file/d/1PmEbaA0y3QXj_sDPVDTarp36LoiVq7Kd/view?usp=sharing) |
| **Deraining** | [MIR-L Derain](https://drive.google.com/file/d/1HuQ-dp25EIeQDfzf4YdP03s0pQv9np9X/view?usp=sharing) |
| **Dehazing** | [MIR-L Dehaze](https://drive.google.com/file/d/1YMHVuoh0TM7hERAgMWQG0Db9MYM-JCVO/view?usp=sharing) |
| **All-in-One** | [MIR-L All-in-One](https://drive.google.com/file/d/12-8y8HkzCEm58e6wdInlzF1WLYDivciz/view?usp=sharing) |

All models are also available on [🤗 Hugging Face](https://huggingface.co/Tomk187/MIR-L).

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
