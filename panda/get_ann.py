# -*- coding: utf-8 -*-
"""
Copyright 2021 huhui, Inc. All Rights Reserved.
@author: huhui
@software: PyCharm
@project: keras-yolo3	
@file: get_ann.py
@version: v1.0
@time: 2021/3/2 下午10:58
@setting: 
-------------------------------------------------
Description :
工程文件说明： 
"""

import os
import glob
import json

img_dirs = ['/data/datasets/PANDA/panda_round1_train_202104_part1',
            '/data/datasets/PANDA/panda_round1_train_202104_part2']
ann_dir = '/data/datasets/PANDA/panda_round1_train_annos_202104'
out_ann_file = 'src_annotations.txt'
person_ann_file = os.path.join(ann_dir, 'person_bbox_train.json')
vehicle_ann_file = os.path.join(ann_dir, 'vehicle_bbox_train.json')
person_ann = json.load(open(person_ann_file, 'r'))
vehicle_ann = json.load(open(vehicle_ann_file, 'r'))
f = open(out_ann_file, 'w')
for path, value in person_ann.items():
    if int(path.split('_')[0]) <= 8:
        img_dir = img_dirs[0]
    else:
        img_dir = img_dirs[1]
    f.write(os.path.join(img_dir, path) + '\t')
    image_id = value['image id']
    image_size = value['image size']
    height, width = image_size['height'], image_size['width']
    objects_list = value['objects list']
    for obj_dict in objects_list:
        category = obj_dict['category']
        if category != 'person':
            continue
        rects = obj_dict['rects']

        head = rects['head']
        visible_body = rects['visible body']
        full_body = rects['full body']
        f.write('{},{},{},{},0\t{},{},{},{},1\t{},{},{},{},2\t'.format(
            int(float(visible_body['tl']['x']) * width),
            int(float(visible_body['tl']['y']) * height),
            int(float(visible_body['br']['x']) * width),
            int(float(visible_body['br']['y']) * height),
            int(float(full_body['tl']['x']) * width),
            int(float(full_body['tl']['y']) * height),
            int(float(full_body['br']['x']) * width),
            int(float(full_body['br']['y']) * height),
            int(float(head['tl']['x']) * width),
            int(float(head['tl']['y']) * height),
            int(float(head['br']['x']) * width),
            int(float(head['br']['y']) * height),
        ))

    vehicle_value = vehicle_ann[path]
    vehicle_objects_list = vehicle_value['objects list']
    for obj_dict in vehicle_objects_list:
        category = obj_dict['category']
        if category == 'vehicles' or category == 'unsure':
            continue
        rect = obj_dict['rect']
        f.write('{},{},{},{},3\t'.format(
            int(float(rect['tl']['x']) * width),
            int(float(rect['tl']['y']) * height),
            int(float(rect['br']['x']) * width),
            int(float(rect['br']['y']) * height),
        ))

    f.write('\n')
f.close()