import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import argparse


def xml_to_csv(xmls_dir):
    xml_list = []
    for xml_file in glob.glob(xmls_dir + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height',
                   'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def xml_csv_save(xmls_dir,save_dir):
    xml_df = xml_to_csv(xmls_dir)
    xml_df.to_csv(save_dir)
    print('Successfully converted xml to csv.')

