# -*- coding: utf-8 -*-
"""
Copyright 2020 huhui, Inc. All Rights Reserved.
@author: huhui
@software: PyCharm	
@project: keras-yolo3	
@file: crop_img.py	
@version: v1.0
@time: 2020/8/9 下午8:38
@setting: 
-------------------------------------------------
Description :
工程文件说明： 
"""

import csv
import os
import numpy as np
import cv2


def get_crop_imgs():
    img_dir = '/data/datasets/df/train_dataset'
    label_path = '/data/datasets/df/train_labels.csv'
    out_img_dir = '/data/datasets/df/crop_train_dataset'
    out_annotations = 'crop_annotations.txt'
    f_label = open(label_path, 'r')
    reader = csv.reader(f_label)
    last_name = None
    ann_dict = dict()
    for i, line in enumerate(reader):
        if i == 0:
            continue
        img_name = line[0]
        if img_name != last_name:
            ann_dict[img_name] = []
            last_name = img_name
        label = line[1]
        ann_dict[img_name].append(list(map(int, label.split(' '))))
    f_label.close()

    np.random.seed(999)
    input_size = (416, 416)
    crop_num = 10

    f_out = open(out_annotations, 'w')

    for img_name, boxes in ann_dict.items():
        # if img_name != 'DBA90206.jpg':
        #     continue
        img = cv2.imread(os.path.join(img_dir, img_name))
        h, w, c = img.shape
        for i in range(crop_num):
            start_h = np.random.choice(range(h - input_size[0]))
            start_w = np.random.choice(range(w - input_size[1]))
            crop_img = img[start_h:start_h + input_size[0], start_w:start_w + input_size[1]]
            label = ''
            for box in boxes:
                ul = (max(box[0], start_w), max(box[1], start_h))  # (x1, y1)
                rb = (min(box[2], start_w + input_size[1]), min(box[3], start_h + input_size[0]))  # (x2, y2)
                if ul[0] >= rb[0] or ul[1] >= rb[1]:
                    continue
                else:
                    x1 = ul[0] - start_w
                    y1 = ul[1] - start_h
                    x2 = rb[0] - start_w
                    y2 = rb[1] - start_h
                    ori_area = (box[2] - box[0]) * (box[3] - box[1])
                    out_area = (x2 - x1) * (y2 - y1)
                    if out_area < ori_area * 0.1:
                        continue
                    else:
                        label += '{},{},{},{},0 '.format(x1, y1, x2, y2)
            if len(label) == 0:
                continue
            else:
                current_img_name = '{}_{}.jpg'.format(img_name.split('.')[0], i + 1)
                out_img_path = os.path.join(out_img_dir, current_img_name)
                cv2.imwrite(out_img_path, crop_img)
                f_out.write('{} {}\n'.format(out_img_path, label))

    f_out.close()


def draw_anno():
    ann_path = 'crop_annotations.txt'
    f = open(ann_path, 'r')
    for line in f.readlines():
        contents = line.strip().split(' ')
        img_path = contents[0]
        img = cv2.imread(img_path)
        for label in contents[1:]:
            x1, y1, x2, y2, _ = list(map(int, label.split(',')))
            img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite('/data/datasets/df/crop_train_dataset/tmp/{}'.format(os.path.basename(img_path)), img)
    f.close()


if __name__ == '__main__':
    # draw_anno()
    get_crop_imgs()