# -*- coding: utf-8 -*-
"""
Copyright 2020 huhui, Inc. All Rights Reserved.
@author: huhui
@software: PyCharm	
@project: keras-yolo3-master	
@file: get_annotations.py	
@version: v1.0
@time: 2020/5/29 下午8:59
@setting: 
-------------------------------------------------
Description :
工程文件说明： 
"""

import csv
import os

img_dir = '/data/datasets/df/train_dataset'

label_path = '/data/datasets/df/train_labels.csv'
out_file = '/home/huhui/Documents/open_sources/object_detection/keras-yolo3/df/annotations.txt'
f_label = open(label_path, 'r')
reader = csv.reader(f_label)
f_out = open(out_file, 'w')
last_name = None
for i, line in enumerate(reader):
    if i == 0:
        continue
    img_name = line[0]
    if img_name != last_name:
        if last_name is not None:
            f_out.write('\n')
        last_name = img_name
        f_out.write('{} '.format(os.path.join(img_dir, img_name)))
    label = line[1]
    f_out.write('{},0 '.format(','.join(label.split(' '))))

f_label.close()
f_out.close()