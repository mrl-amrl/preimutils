import xml.etree.ElementTree as ET
import glob
import os
import cv2
from tqdm import tqdm


def export_image_path(xml_path: str, images_dir: str) -> str:
    """Export image path from the xml file 
        if your image and xml names are same

    Args:
        xml_path: single xml file path.
        images_dir: your images path.

    Returns:
        The return value. True for success, False otherwise.

    """

    if not os.path.exists(xml_path):
        print('file_not_exist')
    file_base = os.path.basename(xml_path)[:-4]
    image_path = glob.glob(os.path.join(images_dir,'{}.jpg'.format(file_base)))[0]
    return image_path


def export_bbox_with_object_name(object_name, xml_path):
    """Export special object name bboxs from the xml file 

    Args:
        object_name: object name that you want to find in your image.
        xml_path: xml of the file that you want to find the object.

    Returns:
        bboxs point from object
    """
    bboxs = []
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for objects in root.findall('object'):
        if objects[0].text == object_name:
            bboxs.append([int(objects[4][0].text),
                          int(objects[4][1].text),
                          int(objects[4][2].text),
                          int(objects[4][3].text)])
    return bboxs


def cut_image(image_path, dst_save, x, y, x_max, y_max, file_number=0):
    """cut image from the bounding box points and write it

    Args:
        image_path: source image path.
        dst_save: destination for saving image file.
        file_number: image write in this pattern croped{file_number}_{base_name}.jpg
            when you have more than one bbox in your xml file you need it
        x,y,x_max_y_max : standard of voc format to crop
    Returns:
        no return
    """
    img = cv2.imread(image_path)
    base_name = os.path.basename(image_path).split('.')[0]
    roi_img = img[y:y_max, x:x_max]
    dst_img = os.path.join(
        dst_save, 'croped{}_{}.jpg'.format(file_number, base_name))
    cv2.imwrite(dst_img, roi_img)


def cut_with_object_names(images_dir, xmls_dir,dst_save, labels):
    """cut images from the bounding box points and write it with their object

    Args:
        images_dir: source image path.
        xmls_dir : source of xmls(annotations).
        dst_save: destination for saving image file.
        labels -> list: labels list 
    Returns:
        no return
    """
    count = 0
    for label in labels:
        if not os.path.exists(os.path.join(dst_save, label)):
            os.makedirs(os.path.join(dst_save, label))

    for xml_path in tqdm(glob.glob(os.path.join(xmls_dir, '*'))):
        image_path = export_image_path(xml_path, images_dir)
        for label in labels:
            bboxs = export_bbox_with_object_name(label, xml_path)
            for bbox in bboxs:
                x, y, x_max, y_max = bbox
                current_dst_save = os.path.join(dst_save, label)
                if not os.path.exists:
                    os.makedirs(current_dst_save)
                cut_image(image_path, current_dst_save,
                          x, y, x_max, y_max, count)
                count += 1
