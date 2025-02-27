from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import os
from models.MIRLModel import MIRLModel
from utils.dataset_utils import TestDataset
from utils.val_utils import Metrics
from utils.image_utils import save_image_tensor
from options.test_options import options as opt
import numpy as np
from tabulate import tabulate

def main():
    np.random.seed(0)
    torch.manual_seed(0)

    ckpt_path = os.path.join(opt.ckpt_dir, opt.ckpt_name)

    denoise_datasets = opt.denoise_test_datasets
    derain_datasets = opt.derain_test_datasets
    dehaze_datasets = opt.dehaze_test_datasets

    # Ensure all modes are run if all-in-one is specified
    if "all_in_one" in opt.mode:
        opt.mode = ["denoise_15", "denoise_25", "denoise_50", "derain", "dehaze"]
    # Ensure all denoise modes are run if denoise is specified
    if "denoise" in opt.mode:
        opt.mode = [mode for mode in opt.mode if mode != "denoise"]
        opt.mode.extend(["denoise_15", "denoise_25", "denoise_50"])

    print("Checkpoint name being loaded: {}\n".format(opt.ckpt_name))

    # Load the checkpoint
    checkpoint = torch.load(ckpt_path, map_location="cuda", weights_only=False)

    # Initialize the model
    net = MIRLModel(opt)
    net.load_state_dict(checkpoint['state_dict'])
    total_params = sum(p.numel() for p in net.net.parameters())
    non_zero_params = sum((p != 0).sum().item() for p in net.net.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")
    print(f"Non-zero trainable parameters: {non_zero_params}")
    print(f"Compression rate : {total_params/non_zero_params:.2f}x ({100 * (total_params-non_zero_params) / total_params:.2f}% pruned)")
    
    net = net.cuda()
    net.eval()

    results = []

    if "denoise_15" in opt.mode:
        for dataset in denoise_datasets:
            print("-" * 80 + "\n" + "-" * 80)
            denoise_testset = TestDataset(opt, "denoise_15", dataset)
            output_path = os.path.join(opt.output_dir, "denoise", dataset, "sigma15")
            os.makedirs(output_path, exist_ok=True)

            print(f'\n======== Testing Denoising with Sigma = 15 on the {dataset} dataset... ========')

            psnr, ssim = test_denoise(net, denoise_testset, dataset, output_path, 15)
            results.append(["Denoise (Sigma=15)", dataset, psnr, ssim])

    if "denoise_25" in opt.mode:
        for dataset in denoise_datasets:
            print("-" * 80 + "\n" + "-" * 80)
            denoise_testset = TestDataset(opt, "denoise_25", dataset)
            output_path = os.path.join(opt.output_dir, "denoise", dataset, "sigma25")
            os.makedirs(output_path, exist_ok=True)

            print(f'\n======== Testing Denoising with Sigma = 25 on the {dataset} dataset... ========')

            psnr, ssim = test_denoise(net, denoise_testset, dataset, output_path, 25)
            results.append(["Denoise (Sigma=25)", dataset, psnr, ssim])
    
    if "denoise_50" in opt.mode:
        for dataset in denoise_datasets:
            print("-" * 80 + "\n" + "-" * 80)
            denoise_testset = TestDataset(opt, "denoise_50", dataset)
            output_path = os.path.join(opt.output_dir, "denoise", dataset, "sigma50")
            os.makedirs(output_path, exist_ok=True)

            print(f'\n======== Testing Denoising with Sigma = 50 on the {dataset} dataset... ========')

            psnr, ssim = test_denoise(net, denoise_testset, dataset, output_path, 50)
            results.append(["Denoise (Sigma=50)", dataset, psnr, ssim])

    if "derain" in opt.mode:
        for dataset in derain_datasets:
            print("-" * 80 + "\n" + "-" * 80)
            derain_testset = TestDataset(opt, "derain", dataset)
            output_path = os.path.join(opt.output_dir, "derain", dataset)
            os.makedirs(output_path, exist_ok=True)

            print(f'\n======== Testing Deraining on the {dataset} dataset... ========')

            psnr, ssim = test_derain(net, derain_testset, dataset, output_path)
            results.append(["Derain", dataset, psnr, ssim])

    if "dehaze" in opt.mode:
        for dataset in dehaze_datasets:
            print("-" * 80 + "\n" + "-" * 80)
            dehaze_testset = TestDataset(opt, "dehaze", dataset)
            output_path = os.path.join(opt.output_dir, "dehaze", dataset)
            os.makedirs(output_path, exist_ok=True)

            print(f'\n======== Testing Dehazing on the {dataset} dataset... ========')
            
            psnr, ssim = test_dehaze(net, dehaze_testset, dataset, output_path)
            results.append(["Dehaze", dataset, psnr, ssim])

    print("\n" + "=" * 80)
    print("Final Results")
    print("=" * 80)
    headers = ["Task", "Dataset", "PSNR", "SSIM"]
    print(tabulate(results, headers=headers, tablefmt="grid"))

def test_denoise(net, dataset, dataset_name, output_path, sigma=None):
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=opt.num_workers)

    metrics = Metrics()

    with torch.no_grad():
        for ([clean_name], degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()

            restored = net(degrad_patch)
            metrics.compute_and_update_psnr_ssim(restored, clean_patch)

            save_image_tensor(restored, os.path.join(output_path, clean_name[0] + '.png'))

        print(f"\n======== Denoising with Sigma = {sigma} Results ({dataset_name} dataset) ========\nPSNR: %.2f         SSIM: %.4f\n\n" % (metrics.psnr_avg, metrics.ssim_avg))

    return float(f"{metrics.psnr_avg:.2f}"), float(f"{metrics.ssim_avg:.4f}")


def test_derain(net, dataset, dataset_name, output_path):
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=opt.num_workers)

    metrics = Metrics()

    with torch.no_grad():
        for ([degraded_name], degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()

            restored = net(degrad_patch)
            metrics.compute_and_update_psnr_ssim(restored, clean_patch)

            save_image_tensor(restored, os.path.join(output_path, degraded_name[0] + '.png'))

        print(f"\n======== Deraining Results ({dataset_name} dataset) ========\nPSNR: %.2f         SSIM: %.4f\n\n" % (metrics.psnr_avg, metrics.ssim_avg))

    return float(f"{metrics.psnr_avg:.2f}"), float(f"{metrics.ssim_avg:.4f}")

def test_dehaze(net, dataset, dataset_name, output_path):
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=opt.num_workers)

    metrics = Metrics()

    with torch.no_grad():
        for ([degraded_name], degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()

            restored = net(degrad_patch)
            metrics.compute_and_update_psnr_ssim(restored, clean_patch)

            save_image_tensor(restored, os.path.join(output_path, degraded_name[0] + '.png'))

        print(f"\n======== Dehazing Results ({dataset_name} dataset) ========\nPSNR: %.2f         SSIM: %.4f\n\n" % (metrics.psnr_avg, metrics.ssim_avg))

    return float(f"{metrics.psnr_avg:.2f}"), float(f"{metrics.ssim_avg:.4f}")


if __name__ == '__main__':
    main()
