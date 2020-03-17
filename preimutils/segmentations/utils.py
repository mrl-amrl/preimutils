from dataset_info import Dataset
from glob import glob
import os
import cv2
from tqdm import tqdm


def find_image_from_mask(mask, images_dir):
    """Export image path from the mask file 
    if your image and xml names are same

    Args:
        mask: single mask.png file path.
        images_dir: your images path.

    Returns:
        image path : path of the input mask  -> string.
    """

    mask = os.path.basename(mask)
    image_path = glob(os.path.join(images_dir, mask[:-4]+'*'))[0]
    return image_path


def find_maxmin_size_images(images_dir):
    """Export the maximum and minimum size of the dataset images 
    if your image and xml names are same

    Args:
        images_dir: your images path.

    Returns:
        image path : path of the input mask  -> string.
    """
    min_height = 10000
    min_width = 10000
    max_height = 0
    max_width = 0
    for image in tqdm(glob(os.path.join(images_dir, '*.*'))):
        img = cv2.imread(image)
        try:
            height, width , _ = img.shape
        except AttributeError:
            print('{} is not a image file ',image)
        min_height = min(min_height, height)
        min_width = min(min_width, width)
        max_height = max(height, max_height)
        max_width = max(width, max_width)

    return {
        'min_height': min_height,
        'min_width': min_width,
        'max_height': max_height,
        'max_width': max_width
    }
