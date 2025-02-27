import os
import random
import copy
from PIL import Image
import numpy as np

from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor
import torch

from utils.image_utils import random_augmentation, crop_img, crop_patch
from utils.degradation_utils import Degradation

class TrainDataset(Dataset):
    """
    Custom Dataset for training models on degraded images with clean targets.
    Supports various degradation types: denoise, derain, and dehaze.
    """

    def __init__(self, args):
        """
        Initialize the dataset with the given arguments.
        Args:
            args: A namespace containing arguments such as directory paths and patch size.
        """
        super(TrainDataset, self).__init__()
        self.args = args
        self.de_type = self.args.mode  # List of degradation types
        self.degrader = Degradation(args)  # Placeholder for Degradation object
        self.sample_ids = []  # Stores the image paths and degradation type

        # Prepare data paths
        self._init_ids()
        self._merge_ids()

        # Define transforms
        self.crop_transform = Compose([
            ToPILImage(),
            RandomCrop(args.patch_size)
        ])
        self.toTensor = ToTensor()

    def _init_ids(self):
        """
        Initialize image paths for each degradation type.
        Reads images from the provided directories and matches clean (target) and degraded (input) images.
        """
        if 'denoise_15' in self.de_type or 'denoise_25' in self.de_type or 'denoise_50' in self.de_type:
            self._init_noisy_ids()
        if 'derain' in self.de_type:
            self._init_rainy_ids()
        if 'dehaze' in self.de_type:
            self._init_hazy_ids()

        # Shuffle degradation types for variability
        random.shuffle(self.de_type)

    def _init_noisy_ids(self):
        """
        Initialize clean (denoising) image paths.
        Matches images from the denoise directory's target and input folders.
        """
        clean_ids = []
        self.s15_ids = []
        self.s25_ids = []
        self.s50_ids = []

        for dataset in self.args.denoise_train_datasets:
            target_dir = os.path.join(self.args.denoise_train_dir, dataset)

            for filename in os.listdir(target_dir):
                if not filename.endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    continue
                clean_ids.append(os.path.join(target_dir, filename))

        if 'denoise_15' in self.de_type:
            self.s15_ids = [{"clean_id": x, "de_type":0} for x in clean_ids]

            # Data augmentation
            if self.args.denoise_data_augmentation < 1:
                self.s15_ids = self.s15_ids
            else:
                self.s15_ids = self.s15_ids * self.args.denoise_data_augmentation

            random.shuffle(self.s15_ids)
        if 'denoise_25' in self.de_type:
            self.s25_ids = [{"clean_id": x, "de_type":1} for x in clean_ids]
            
            # Data augmentation
            if self.args.denoise_data_augmentation < 1:
                self.s25_ids = self.s25_ids
            else:
                self.s25_ids = self.s25_ids * self.args.denoise_data_augmentation

            random.shuffle(self.s25_ids)
        if 'denoise_50' in self.de_type:
            self.s50_ids = [{"clean_id": x, "de_type":2} for x in clean_ids]
            
            # Data augmentation
            if self.args.denoise_data_augmentation < 1:
                self.s50_ids = self.s50_ids
            else:
                self.s50_ids = self.s50_ids * self.args.denoise_data_augmentation            

            random.shuffle(self.s50_ids)

        print(f"Total Denoise Ids: {len(clean_ids)}")

    def _init_rainy_ids(self):
        """
        Initialize deraining image paths.
        Matches images from the derain directory's target and input folders.
        """

        rainy_ids = []

        for dataset in self.args.derain_train_datasets:
            target_dir = os.path.join(self.args.derain_train_dir, dataset, "target")
            input_dir = os.path.join(self.args.derain_train_dir, dataset, "input")

            
            for filename in os.listdir(input_dir):
                if not filename.endswith(("_rainy.jpg", "_rainy.jpeg", "_rainy.png", "_rainy.bmp")):
                    continue
                base_name = filename.rsplit("_rainy", 1)[0]
                extensions = [".jpg", ".jpeg", ".png", ".bmp"]
                for ext in extensions:
                    clean_path = os.path.join(target_dir, base_name + "_gt" + ext)
                    rainy_path = os.path.join(input_dir, filename)
                    if os.path.exists(clean_path):
                        rainy_ids.append({"degrad_id": rainy_path, "clean_id": clean_path, "de_type": 4})
                        break

        self.rainy_ids = rainy_ids

        # Data augmentation
        if self.args.derain_data_augmentation < 1:
            self.rainy_ids = self.rainy_ids
        else:
            self.rainy_ids = self.rainy_ids * self.args.derain_data_augmentation

        random.shuffle(self.rainy_ids)

        print(f"Total Rainy Ids: {len(rainy_ids)}")

    def _init_hazy_ids(self):
        """
        Initialize dehazing image paths.
        Matches images from the dehaze directory's target and input folders.
        """

        hazy_ids = []

        for dataset in self.args.dehaze_train_datasets:
            target_dir = os.path.join(self.args.dehaze_train_dir, dataset, "target")
            input_dir = os.path.join(self.args.dehaze_train_dir, dataset, "input")

            for filename in os.listdir(input_dir):
                if not filename.endswith(("_hazy.jpg", "_hazy.jpeg", "_hazy.png", "_hazy.bmp")):
                    continue
                base_name = filename.rsplit("_hazy", 1)[0]
                extensions = [".jpg", ".jpeg", ".png", ".bmp"]
                for ext in extensions:
                    clean_path = os.path.join(target_dir, base_name + "_gt" + ext)
                    hazy_path = os.path.join(input_dir, filename)
                    if os.path.exists(clean_path):
                        hazy_ids.append({"degrad_id": hazy_path, "clean_id": clean_path, "de_type": 4})
                        break

        self.hazy_ids = hazy_ids

        # Data augmentation
        if self.args.dehaze_data_augmentation < 1:
            self.hazy_ids = self.hazy_ids
        else:
            self.hazy_ids = self.hazy_ids * self.args.dehaze_data_augmentation

        random.shuffle(self.hazy_ids)
        print(f"Total Hazy Ids: {len(hazy_ids)}")

    def _merge_ids(self):
        """
        Merge all image IDs into a single list based on the specified degradation types.
        """
        self.sample_ids = []
        if "denoise_15" in self.de_type:
            self.sample_ids += self.s15_ids
        if "denoise_25" in self.de_type:
            self.sample_ids += self.s25_ids
        if "denoise_50" in self.de_type:
            self.sample_ids += self.s50_ids
        if "derain" in self.de_type:
            self.sample_ids+= self.rainy_ids
        if "dehaze" in self.de_type:
            self.sample_ids+= self.hazy_ids

        print(f"Total Sample Ids: {len(self.sample_ids)}")

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        Args:
            idx: Index of the sample.
        Returns:
            Tuple of metadata, degraded image, and clean image.
        """
        sample = self.sample_ids[idx]
        de_id = sample["de_type"]

        if de_id < 3:  # Denoising
            clean_id = sample["clean_id"]
            clean_img = np.array(Image.open(clean_id).convert('RGB'))
            clean_patch = self.crop_transform(clean_img)
            clean_patch = np.array(clean_patch)
            clean_patch = random_augmentation(clean_patch)[0]

            if de_id == 0:
                degrad_patch = self.degrader.degrade(clean_patch, 15)
            elif de_id == 1:
                degrad_patch = self.degrader.degrade(clean_patch, 25)
            elif de_id == 2:
                degrad_patch = self.degrader.degrade(clean_patch, 50)

        else:  # Deraining or Dehazing
            clean_id = sample["clean_id"]
            degraded_id = sample["degrad_id"]
            degraded_path = degraded_id
            degrad_img = np.array(Image.open(degraded_path).convert('RGB'))
            clean_img = np.array(Image.open(clean_id).convert('RGB'))

            degrad_patch, clean_patch = random_augmentation(*crop_patch(degrad_img, clean_img, self.args.patch_size))

        return [clean_id, de_id], self.toTensor(degrad_patch), self.toTensor(clean_patch)

    def __len__(self):
        """
        Return the total number of samples.
        """
        return len(self.sample_ids)


class TestDataset(Dataset):
    """
    Unified dataset class for denoising, deraining, and dehazing tasks.
    """
    def __init__(self, args, task=None, dataset_name=None):
        super(TestDataset, self).__init__()
        self.args = args
        self.de_type = task  # Degradation type
        self.degrader = Degradation(args)  # Placeholder for Degradation object
        self.dataset_name = dataset_name # Name of the dataset to load

        # Set sigma if task is denoising
        if self.de_type == "denoise_15" or self.de_type == "denoise_25" or self.de_type == "denoise_50":
            self.sigma = int(self.de_type.split("_")[1])

        self._init_ids()

        self.toTensor = ToTensor()

    def _init_ids(self):
        """
        Initialize image IDs based on the selected task.
        """
        if self.de_type == 'denoise_15' or self.de_type == 'denoise_25' or self.de_type == 'denoise_50':
            noisy_ids = []

            target_dir = os.path.join(self.args.denoise_test_dir, self.dataset_name)

            for filename in os.listdir(target_dir):
                if not filename.endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    continue
                noisy_ids.append({"clean_id": os.path.join(target_dir, filename)})

            self.noisy_ids = noisy_ids
            self.length = len(self.noisy_ids)

        elif self.de_type == 'derain':
            rainy_ids = []

            target_dir = os.path.join(self.args.derain_test_dir, self.dataset_name, "target")
            input_dir = os.path.join(self.args.derain_test_dir, self.dataset_name, "input")

            for filename in os.listdir(input_dir):
                if not filename.endswith(("_rainy.jpg", "_rainy.jpeg", "_rainy.png", "_rainy.bmp")):
                    continue
                base_name = filename.rsplit("_rainy", 1)[0]
                extensions = [".jpg", ".jpeg", ".png", ".bmp"]
                for ext in extensions:
                    clean_path = os.path.join(target_dir, base_name + "_gt" + ext)
                    rainy_path = os.path.join(input_dir, filename)
                    if os.path.exists(clean_path):
                        rainy_ids.append({"degrad_id": rainy_path, "clean_id": clean_path})
                        break

            self.rainy_ids = rainy_ids
            self.length = len(self.rainy_ids)

        elif self.de_type == 'dehaze':
            hazy_ids = []

            target_dir = os.path.join(self.args.dehaze_test_dir, self.dataset_name, "target")
            input_dir = os.path.join(self.args.dehaze_test_dir, self.dataset_name, "input")

            for filename in os.listdir(input_dir):
                if not filename.endswith(("_hazy.jpg", "_hazy.jpeg", "_hazy.png", "_hazy.bmp")):
                    continue
                base_name = filename.rsplit("_hazy", 1)[0]
                extensions = [".jpg", ".jpeg", ".png", ".bmp"]
                for ext in extensions:
                    clean_path = os.path.join(target_dir, base_name + "_gt" + ext)
                    hazy_path = os.path.join(input_dir, filename)
                    if os.path.exists(clean_path):
                        hazy_ids.append({"degrad_id": hazy_path, "clean_id": clean_path})
                        break
                            
            self.hazy_ids = hazy_ids
            self.length = len(self.hazy_ids)

    def _add_gaussian_noise(self, clean_patch):
        """
        Add Gaussian noise to the clean image patch.
        """
        noise = np.random.randn(*clean_patch.shape)
        noisy_patch = np.clip(clean_patch + noise * self.sigma, 0, 255).astype(np.uint8)
        return noisy_patch

    def __getitem__(self, idx):
        """
        Get a data sample for the specified task.
        """
        if self.de_type == 'denoise_15' or self.de_type == 'denoise_25' or self.de_type == 'denoise_50':  # Denoising
            clean_img_path = self.noisy_ids[idx]["clean_id"]

            clean_img = crop_img(np.array(Image.open(clean_img_path).convert('RGB')), base=16)
            noisy_img = self.degrader.degrade(clean_img, self.sigma)

            clean_img, noisy_img = self.toTensor(clean_img), self.toTensor(noisy_img)
            clean_name = os.path.basename(clean_img_path).split('.')[0]

            return [clean_name], noisy_img, clean_img

        else:  # Deraining or Dehazing
            degraded_img_path = self.rainy_ids[idx]["degrad_id"] if self.de_type == "derain" else self.hazy_ids[idx]["degrad_id"]
            clean_img_path = self.rainy_ids[idx]["clean_id"] if self.de_type == "derain" else self.hazy_ids[idx]["clean_id"]

            degraded_img = crop_img(np.array(Image.open(degraded_img_path).convert('RGB')), base=16)
            clean_img = crop_img(np.array(Image.open(clean_img_path).convert('RGB')), base=16)

            clean_img, degraded_img = self.toTensor(clean_img), self.toTensor(degraded_img)
            degraded_name = os.path.basename(degraded_img_path).split('.')[0]

            return [degraded_name], degraded_img, clean_img

    def __len__(self):
        """
        Return the length of the dataset.
        """
        return self.length