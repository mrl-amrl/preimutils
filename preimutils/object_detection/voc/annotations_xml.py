import cv2
import xml.etree.ElementTree as ET
import os
import pascal_voc_writer as pascal
from .crop_from_point import export_image_path

class AnnotationsXML:
    def __init__(self, images_dir, xmls_dir):
        self.images_dir = images_dir
        self.xmls_dir = xmls_dir

    def parse_pascal_voc(self, xml_file_path, categori_label_id):
        imagepath_with_bounding_box = {}
        bboxs = []
        class_names = []
        category_id = []
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        imagepath_with_bounding_box['image_path'] = export_image_path(xml_file_path,self.images_dir)
        for objects in root.findall('object'):
            class_names.append(objects.findall('name')[0].text)
            bbox = objects.findall('bndbox')[0]
            bboxs.append([int(bbox.findall('xmin')[0].text),
                          int(bbox.findall('ymin')[0].text),
                          int(bbox.findall('xmax')[0].text),
                          int(bbox.findall('ymax')[0].text)])

            category_id.append(categori_label_id[objects[0].text])

        imagepath_with_bounding_box['class_names'] = class_names
        imagepath_with_bounding_box['category_id'] = category_id
        imagepath_with_bounding_box['bboxes'] = bboxs
        image = cv2.imread(
            imagepath_with_bounding_box['image_path'], cv2.IMREAD_COLOR)
        imagepath_with_bounding_box['image'] = image
        return imagepath_with_bounding_box

    def pascal_image_voc_writer(self, annotations, image_xml_name, categori_id_label):
        img = annotations['image']
        img_path = os.path.join(self.images_dir, image_xml_name + '.jpg')
        cv2.imwrite(img_path, img)
        height, width, _ = img.shape
        Writer = pascal.Writer(img_path, width, height)
        for idx, bbox in enumerate(annotations['bboxes']):
            label_text = categori_id_label[annotations['category_id'][idx]]
            x_min, y_min, x_max, y_max = bbox
            x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
            Writer.addObject(name = str(label_text), xmin = x_min, ymin = y_min, xmax = x_max, ymax= y_max)
        Writer.save(os.path.join(self.xmls_dir, image_xml_name + '.xml'))


        

