from dataset_info import Dataset
from glob import glob
import os


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

    for image in glob(os.path.join(images_dir):
        pass
