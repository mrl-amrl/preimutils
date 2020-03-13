from albumentations import *
import cv2

def aug(image,mask):
    # aug = HorizontalFlip(p=1)
    aug = Resize(500,500)
    image = cv2.imread(image)
    mask = cv2.imread(mask)
    augmented = aug(image=image, mask=mask)
    image_h_flipped = augmented['image']
    mask_h_flipped = augmented['mask']
    cv2.imshow('aug',image_h_flipped)
    cv2.imshow('mask',mask_h_flipped)
    cv2.waitKey(0)

