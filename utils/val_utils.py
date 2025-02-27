import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

class Metrics():
    """ Computes and stores various metrics, along with their total, average and latest value """

    def __init__(self):
        self._reset()

    def _reset(self):
        """ Reset all metrics """
        self.psnr = 0
        self.psnr_sum = 0
        self.psnr_avg = 0
        self.psnr_count = 0
        self.ssim = 0
        self.ssim_sum = 0
        self.ssim_avg = 0
        self.ssim_count = 0

    def _update(self, metric_name, value, n=None):
        """
        Update metrics with the given value and count (n).

        Parameters:
        metric_name (str): Name of the metric to update ('psnr' or 'ssim')
        value (float): Latest value of the metric
        n (int): Number of samples for the value (default is 1)
        """
        if metric_name == 'psnr':
            self.psnr = value  # Latest PSNR value
            self.psnr_sum += value * n  # Update sum
            self.psnr_count += n  # Update count
            self.psnr_avg = self.psnr_sum / self.psnr_count  # Compute average

        elif metric_name == 'ssim':
            self.ssim = value  # Latest SSIM value
            self.ssim_sum += value * n  # Update sum
            self.ssim_count += n  # Update count
            self.ssim_avg = self.ssim_sum / self.ssim_count  # Compute average

    def _compute_psnr(self, restored, clean):
        assert restored.shape == clean.shape
        restored = np.clip(restored.detach().cpu().numpy(), 0, 1)
        clean = np.clip(clean.detach().cpu().numpy(), 0, 1)

        restored = restored.transpose(0, 2, 3, 1)
        clean = clean.transpose(0, 2, 3, 1)
        psnr = 0

        for i in range(restored.shape[0]):
            psnr += peak_signal_noise_ratio(clean[i], restored[i], data_range=1)

        return psnr / restored.shape[0], restored.shape[0]

    def _compute_ssim(self, restored, clean):
        assert restored.shape == clean.shape
        restored = np.clip(restored.detach().cpu().numpy(), 0, 1)
        clean = np.clip(clean.detach().cpu().numpy(), 0, 1)

        restored = restored.transpose(0, 2, 3, 1)
        clean = clean.transpose(0, 2, 3, 1)
        ssim = 0

        for i in range(restored.shape[0]):
            ssim += structural_similarity(clean[i], restored[i], data_range=1, channel_axis=-1)

        return ssim / restored.shape[0], restored.shape[0]

    def compute_and_update_psnr_ssim(self, restored, clean):
        psnr, N_psnr = self._compute_psnr(restored, clean)
        ssim, N_ssim = self._compute_ssim(restored, clean)

        self._update('psnr', psnr, N_psnr)
        self._update('ssim', ssim, N_ssim)