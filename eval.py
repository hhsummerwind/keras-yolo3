# -*- coding: utf-8 -*-
"""
Copyright 2020 huhui, Inc. All Rights Reserved.
@author: huhui
@software: PyCharm	
@project: yolo3SaftyHatDuJian	
@file: eval.py	
@version: v1.0
@time: 2020/8/27 上午10:29
@setting: 
-------------------------------------------------
Description :
工程文件说明： 
"""

import os
from PIL import Image

from yolo import YOLO

labels = ['hat', 'person']


def get_gt():
    print("ground truth:-----------------------------")
    val_path = '/data/datasets/tianji/hat/hat_val.txt'
    dst_dir = '/data/datasets/tianji/hat/val_annos'
    f = open(val_path, 'r')
    for line in f.readlines():
        contents = line.strip().split(' ')
        img_path = contents[0]
        print(img_path)
        out_path = os.path.join(dst_dir, '{}.txt'.format(os.path.basename(os.path.splitext(img_path)[0])))
        out_f = open(out_path, 'w')
        for i, anno in enumerate(contents[1:]):
            x1, y1, x2, y2, label = anno.split(',')
            out_f.write('{} {}\n'.format(labels[int(label)], ' '.join([x1, y1, x2, y2])))
        print(i)
        out_f.close()
    f.close()


def get_result():
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
    sess = tf.Session(config=config)
    set_session(tf.Session(config=config))
    print("get result:-----------------------------")
    yolo = YOLO()
    val_path = '/data/datasets/tianji/hat/hat_val.txt'
    dst_dir = '/data/models/hat/input608_box200/result'
    f = open(val_path, 'r')
    for i, line in enumerate(f.readlines()):
        # if i == 107:
        #     break
        contents = line.strip().split(' ')
        img_path = contents[0]
        print(i, img_path)
        image = Image.open(img_path)
        r_image, out_boxes, out_scores, out_classes = yolo.detect_image(image)
        out_path = os.path.join(dst_dir, '{}.txt'.format(os.path.basename(os.path.splitext(img_path)[0])))
        out_f = open(out_path, 'w')
        for box, score, out_class in zip(out_boxes, out_scores, out_classes):
            out_f.write('{} {} {} {} {} {}\n'.format(labels[out_class], score, box[1], box[0], box[3], box[2]))
        out_f.close()
    f.close()


if __name__ == '__main__':
    # get_gt()
    get_result()