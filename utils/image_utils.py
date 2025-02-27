import numpy as np
import random
from PIL import Image

def crop_patch(img_1, img_2, patch_size):
    """
    Randomly crop patches from two aligned images.
    Args:
        img_1: First image (degraded).
        img_2: Second image (clean).
    Returns:
        Cropped patches from both images.
    """
    H, W = img_1.shape[:2]
    ind_H = random.randint(0, H - patch_size)
    ind_W = random.randint(0, W - patch_size)

    patch_1 = img_1[ind_H:ind_H + patch_size, ind_W:ind_W + patch_size]
    patch_2 = img_2[ind_H:ind_H + patch_size, ind_W:ind_W + patch_size]

    return patch_1, patch_2

# Crop an image to the multiple of the given base
def crop_img(image, base=None):
    h = image.shape[0]
    w = image.shape[1]
    crop_h = h % base
    crop_w = w % base
    return image[crop_h // 2:h - crop_h + crop_h // 2, crop_w // 2:w - crop_w + crop_w // 2, :]

# Apply an augmentation to the given image
def _apply_augmentation(image, aug):
    if aug == 0:
        # No augmentation
        out = image.numpy()
    elif aug == 1:
        # Flip up and down
        out = np.flipud(image)
    elif aug == 2:
        # Rotate counterclockwise 90 degrees
        out = np.rot90(image)
    elif aug == 3:
        # Rotate 90 degrees and flip
        out = np.rot90(image)
        out = np.flipud(out)
    elif aug == 4:
        # Rotate 180 degrees
        out = np.rot90(image, k=2)
    elif aug == 5:
        # Rotate 180 degrees and flip
        out = np.rot90(image, k=2)
        out = np.flipud(out)
    elif aug == 6:
        # Rotate 270 degrees
        out = np.rot90(image, k=3)
    elif aug == 7:
        # Rotate 270 degrees and flip
        out = np.rot90(image, k=3)
        out = np.flipud(out)

    return out

# Apply a random augmentation to the given images
def random_augmentation(*images):
    out = []
    augmentation = random.randint(1, 7)
    for img in images:
        out.append(_apply_augmentation(img, augmentation).copy())

    return out

def _torch_to_np(img_var):
    """
    Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    :param img_var:
    :return:
    """
    return img_var.detach().cpu().numpy()[0]

def _np_to_pil(img_np):
    """
    Converts image in np.array format to PIL image.

    From C x W x H [0..1] to  W x H x C [0...255]
    :param img_np:
    :return:
    """
    ar = np.clip(img_np * 255, 0, 255).astype(np.uint8)

    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        assert img_np.shape[0] == 3, img_np.shape
        ar = ar.transpose(1, 2, 0)

    return Image.fromarray(ar)

def save_image_tensor(image_tensor, output_path=None):
    image_np = _torch_to_np(image_tensor)
    p = _np_to_pil(image_np)
    p.save(output_path)