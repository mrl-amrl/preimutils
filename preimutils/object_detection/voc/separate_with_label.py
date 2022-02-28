import os
import glob
import xml.etree.ElementTree as ET
from tqdm import tqdm
from .label_json import LabelHandler
import shutil


def export_path_count_for_each_label(xmls_dir, images_dir, labels):
    """Get statistics of dataset with their labels with their xmls and images files path

    Args:
        xmls_dir: all xmls file directory.
        images_dir: your images directory.
        labels:['object1','object2',...,'objectN']

    Return:
        dict{   label1: {
            count:
            xmls_paths:[]
            images_paths:[]
                    },
                    ...,
        labelN: {
            count:
            xmls_paths:[]
            images_paths:[]
                    }
        }
    """
    labels_statistics = {}
    for label in labels:
        labels_statistics[label] = {'count': 0,
                                    'xmls_paths': [], 'images_paths': []}
    for xml_file in tqdm(glob.glob(os.path.join(xmls_dir, '*.xml')), desc='statistic'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            current_label = member.find('name').text
            xml_base_name = os.path.basename(xml_file)
            file_base_name = xml_base_name.split('.')[0]
            image_path = glob.glob(os.path.join(
                images_dir, '{}.*'.format(file_base_name)))

            labels_statistics[current_label]['count'] += 1
            if xml_file not in labels_statistics[current_label]['xmls_paths']:
                labels_statistics[current_label]['xmls_paths'].append(xml_file)
                labels_statistics[current_label]['images_paths'].append(
                    image_path[0])
    return labels_statistics


def separate_with_label(xmls_dir, images_dir, labels_array):
    """Seprate your dataset with their labels in seprate folder path

    Args:
        xmls_dir: all xml file directory.
        images_dir: your images directory.
        labels_array:['object1','object2',...,'objectN']

    Returns:
        No return
    """
    for label in labels_array:
        label_path = os.path.join(os.path.dirname(images_dir), label)
        if not os.path.exists(label_path):
            os.mkdir(label_path)
        annotations = os.path.join(label_path, 'annotations')
        images = os.path.join(label_path, 'images')
        if not os.path.exists(annotations):
            os.mkdir(annotations)
        if not os.path.exists(images):
            os.mkdir(images)
    for xml_file in tqdm(glob.glob(xmls_dir + '/*.xml')):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            current_label = member.find('name').text
            xml_base_name = os.path.basename(xml_file)
            file_base_name = xml_base_name.split('.')[0]
            image_file = glob.glob(
                images_dir + '/{}.*'.format(file_base_name))
            current_img_name = os.path.basename(*image_file)
            print(*image_file)
            dst_path = os.path.join(
                os.path.dirname(images_dir), current_label)
            # find destinations path
            img_dst_path = os.path.join(os.path.join(
                dst_path, 'images'), current_img_name)
            xml_dst_path = os.path.join(os.path.join(
                dst_path, 'annotations'), xml_base_name)
            # copy images
            shutil.copy2(*image_file, img_dst_path)
            # copy labels
            shutil.copy2(xml_file, xml_dst_path)


def gather_together(labels, dataset_dir):
    """Gather together images and their annotation.Use after doing separate_with_label this method
    Args:
        xmls_dir: all xmls files directory.
        dataset_dir: dataset directory.
        labels_array:['object1','object2',...,'objectN']

    Returns:
        No return
    """

    for label in tqdm(labels):
        label_obj_path = os.path.join(dataset_dir, label)
        images_dir = os.path.join(label_obj_path, 'images')
        xmls_dir = os.path.join(label_obj_path, 'annotations')

        dst_images_path = os.path.join(
            dataset_dir, 'augmented_image', 'images')
        dst_xmls_path = os.path.join(
            dataset_dir, 'augmented_image', 'annotations')
        if not os.path.exists(dst_images_path):
            os.makedirs(dst_images_path)
        if not os.path.exists(dst_xmls_path):
            os.makedirs(dst_xmls_path)
        if not os.path.exists(images_dir):
            print('image path not exist {}'.format(images_dir))
        if not os.path.exists(xmls_dir):
            print('xmls path not exist {}'.format(xmls_dir))
        for source_image in glob.glob(images_dir + '/*'):
            dst_image_path = os.path.join(
                dst_images_path, os.path.basename(source_image))
            shutil.copy2(source_image, dst_image_path)
        for source_xml in glob.glob(xmls_dir+'/*'):
            dst_xml_path = os.path.join(
                dst_xmls_path, os.path.basename(source_xml))
            shutil.copy2(source_xml, dst_xml_path)
