# -*- coding: utf-8 -*-
"""
Copyright 2021 huhui, Inc. All Rights Reserved.
@author: huhui
@software: PyCharm
@project: keras-yolo3	
@file: check_ann.py
@version: v1.0
@time: 2021/3/4 下午9:43
@setting: 
-------------------------------------------------
Description :
工程文件说明： 
"""

import os
import colorsys
import numpy as np
from PIL import Image, ImageFont, ImageDraw
Image.MAX_IMAGE_PIXELS = 933120000

ann_file = 'panda/split2_608_annotations.txt'#'panda/src_annotations.txt'#'panda/split1_annotations.txt'
out_dir = '/data/datasets/PANDA/tmp'
label_path = 'panda/class_names.txt'
file = open(label_path, 'r')
class_names = [line.strip() for line in file.readlines()]
file.close()
hsv_tuples = [(x / len(class_names), 1., 1.)
              for x in range(len(class_names))]
colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
colors = list(
    map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
        colors))

f = open(ann_file, 'r')
for line in f.readlines():
    contents = line.strip().split('\t')
    if len(contents) == 1:
        continue
    img_path = contents[0]
    image = Image.open(img_path)
    draw = ImageDraw.Draw(image)
    for ann in contents[1:]:
        left, top, right, bottom, label_ind = list(map(int, ann.split(',')))
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = max((image.size[0] + image.size[1]) // 300, 1)
        label_size = draw.textsize(class_names[label_ind], font)
        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])
        for i in range(thickness):
            draw.rectangle(
                [left + i, top + i, right - i, bottom - i],
                outline=colors[label_ind])
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill=colors[label_ind])
        draw.text(text_origin, class_names[label_ind], fill=(0, 0, 0), font=font)
    del draw
    image.save(os.path.join(out_dir, os.path.basename(img_path)))
    # break

f.close()