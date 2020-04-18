from dataset_info import Dataset
from glob import glob
import os
import cv2
from tqdm import tqdm
import numpy as np

def encode_segmap(self, mask,class_color):
    """Encode segmentation label images as pascal classes
    Args:
        mask (np.ndarray): raw segmentation label image of dimension
            (M, N, 3), in which the Pascal classes are encoded as colours.
        class_color(list): class colors 
    Returns:
        (np.ndarray): class map with dimensions (M,N), where the value at
        a given location is the integer denoting the class index.
    """
    mask = mask.astype(int)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
    for ii, label in enumerate(class_color):
        label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
    label_mask = label_mask.astype(int)
    return label_mask

def decode_segmap(self, label_mask, plot=False):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
            the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
            in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    label_colours = self.get_pascal_labels()
    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, self.n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb

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
