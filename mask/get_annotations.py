# -*- coding: utf-8 -*-
"""
Copyright 2021 huhui, Inc. All Rights Reserved.
@author: huhui
@software: PyCharm
@project: keras-yolo3	
@file: get_annotations.py
@version: v1.0
@time: 2021/2/9 下午1:42
@setting: 
-------------------------------------------------
Description :
工程文件说明： 
"""

import os
import glob
import xml.etree.ElementTree as ET


classes = ['face', 'face_mask']
def convert_annotation(in_file, list_file):
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
#
# for year, image_set in sets:
#     image_ids = open('/data/datasets/VOC/VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
#     list_file = open('voc/%s_%s.txt'%(year, image_set), 'w')
#     for image_id in image_ids:
#         list_file.write('/data/datasets/VOC/VOCdevkit/VOC%s/JPEGImages/%s.jpg'%(year, image_id))
#         convert_annotation(year, image_id, list_file)
#         list_file.write('\n')
#     list_file.close()


img_dir = '/data/datasets/mask/FaceMaskDataset'
dst_dir = '/home/huhui/Documents/open_sources/object_detection/keras-yolo3/mask'
for stage in os.listdir(img_dir):
    out_ann_path = os.path.join(dst_dir, '{}.txt'.format(stage))
    dir_path = os.path.join(img_dir, stage)
    if not os.path.isdir(dir_path):
        continue
    f = open(out_ann_path, 'w')
    for xml_path in glob.glob(os.path.join(dir_path, '*.xml')):
        img_path = xml_path.replace('.xml', '.jpg')
        if not os.path.exists(img_path):
            continue
        f.write(img_path)
        convert_annotation(xml_path, f)
        f.write('\n')

    f.close()