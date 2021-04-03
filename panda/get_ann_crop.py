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
import json
from collections import defaultdict

ann_file = '/data/datasets/PANDA/crop/panda_round1_coco_full_6categories_val.json'
img_dir = '/data/datasets/PANDA/crop/panda_round1_train_202104_patches_4096_3500'
out_ann_file = 'annotations.txt'
f = open(ann_file, encoding='utf-8')
data = json.load(f)
f.close()
name_box_id = defaultdict(list)

annotations = data['annotations']
infos = data['images']

image_names = dict()
for info in infos:
    image_names.update({info['id']: info['file_name']})

for ant in annotations:
    id = ant['image_id']
    # name = '/data/datasets/coco/train2017/%012d.jpg' % id
    cat = ant['category_id']

    name_box_id[id].append([ant['bbox'], cat])


f = open(out_ann_file, 'w')
for key in name_box_id.keys():
    f.write(os.path.join(img_dir, image_names[key]))
    box_infos = name_box_id[key]
    for info in box_infos:
        x_min = int(info[0][0])
        y_min = int(info[0][1])
        x_max = x_min + int(info[0][2])
        y_max = y_min + int(info[0][3])

        box_info = " %d,%d,%d,%d,%d" % (
            x_min, y_min, x_max, y_max, int(info[1]))
        f.write(box_info)
    f.write('\n')
f.close()
