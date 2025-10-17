## Installation and Requirements

1. Clone this repository:
   ```bash
   git clone https://github.com/Thomkat/MIR-L.git
   cd MIR-L
   ```
2. Create a new conda environment using the file provided:
   ```bash
   conda env create -n mirl -f environment.yml
   conda activate mirl
   ```
3. Activate the environment:
   ```bash
   conda activate mirl
   ```
   
### Important Package Versions

- **Python**: 3.9.21  
- **CUDA**: 12.1  
- **cuDNN**: 9.1.0  
- **PyTorch**: 2.5.1  
- **TorchVision**: 0.20.1  
- **PyTorch Lightning**: 2.4.0 

---

## Dataset Structure

Train/test datasets are available to download [**here**](https://drive.google.com/file/d/1NsDbszHF3OIJk_jMnXg2LuOqE8sXQvU8/view?usp=sharing).

The code expects the following folder structure under `./datasets/`, by default:

```
datasets/
├── train/
│   ├── denoise/
│   │   ├── WED/
│   │   │   ├── 00007.bmp
│   │   │   ├── 00010.bmp
│   │   │   └── ...
│   │   └── BSD400/
│   │       ├── 100080.jpg
│   │       ├── 104055.jpg
│   │       └── ...
│   ├── derain/
│   │   └── Rain100L/
│   │       ├── target/
│   │       │   ├── rain-101_gt.png
│   │       │   ├── rain-108_gt.png
│   │       │   └── ...
│   │       └── input/
│   │           ├── rain-101_rainy.png
│   │           ├── rain-108_rainy.png
│   │           └── ...
│   └── dehaze/
│       └── OTS/
│           ├── target/
│           │   ├── 0025_0.9_0.2_gt.jpg
│           │   ├── 0039_0.95_0.2_gt.jpg
│           │   └── ...
│           └── input/
│               ├── 0025_0.9_0.2_gt.jpg
│               ├── 0039_0.95_0.2_gt.jpg
│               └── ...
└── test/
    ├── denoise/
    │   ├── BSD68/
    │   │   ├── 101085.jpg
    │   │   ├── 101087.jpg    
    │   │   └── ...
    │   └── Urban100/
    │       ├── img_001.png
    │       ├── img_002.png
    │       └── ...
    ├── derain/
    │   └── Rain100L/
    │       ├── target/
    │       │   ├── rain-001_gt.png
    │       │   ├── rain-002_gt.png
    │       │   └── ...
    │       └── input/
    │           ├── rain-001_rainy.png
    │           ├── rain-002_rainy.png
    │           └── ...
    └── dehaze/
        └── SOTS/
            ├── target/
            │   ├── 0001_0.8_0.2_gt.png
            │   ├── 0002_0.8_0.08_gt.png
            │   └── ...
            └── input/
                ├── 0001_0.8_0.2_hazy.jpg
                ├── 0002_0.8_0.08_hazy.jpg
                └── ...
```

### Notes on Dataset Naming Conventions

- **Denoising**: 
  - Training & testing images: The folders contain clean images only. The code will synthetically add Gaussian noise (sigma=15, 25, or 50) on the fly.
- **Deraining**:
  - For training and testing, the `target/` folder has ground-truth images with suffix `_gt`, and the `input/` folder has rainy images with suffix `_rainy`.
- **Dehazing**:
  - For training and testing, the `target/` folder has ground-truth images with suffix `_gt`, and the `input/` folder has hazy images with suffix `_hazy`.

Paths can be customized via command-line arguments (`--denoise_train_dir`, `--derain_train_dir`, `--dehaze_train_dir`, `--denoise_test_dir`, `--derain_test_dir`, `--dehaze_test_dir`) and dataset names via command-line argument(`--denoise_train_datasets`, `--derain_train_datasets`, `--dehaze_train_datasets`, `--denoise_test_datasets`, `--derain_test_datasets`, `--dehaze_test_datasets`)

---