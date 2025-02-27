import random
import numpy as np

class Degradation(object):
    def __init__(self, args):
        super(Degradation, self).__init__()
        self.args = args

    def _add_gaussian_noise(self, clean_patch, sigma):
        noise = np.random.randn(*clean_patch.shape)
        noisy_patch = np.clip(clean_patch + noise * sigma, 0, 255).astype(np.uint8)

        return noisy_patch

    def degrade(self, clean_patch, sigma=None):
        degraded_patch = self._add_gaussian_noise(clean_patch, sigma)

        return degraded_patch