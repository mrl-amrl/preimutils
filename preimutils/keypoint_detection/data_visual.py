import matplotlib.pyplot as plt
import numpy as np


def visualize_keypoints(images, keypoints, save: bool = False, name: str = None):
    image = images
    current_keypoint = keypoints
    plt.imshow(image)
    current_keypoint = np.array(current_keypoint)
    print(len(current_keypoint))
    current_keypoint = current_keypoint[:, :2]
    for idx, (x, y) in enumerate(current_keypoint):
        plt.scatter([x], [y], marker=".", s=50, linewidths=5)
    if save:
        plt.savefig(f'{name}.png')
    plt.figure(figsize=(30, 30))
    plt.show()
