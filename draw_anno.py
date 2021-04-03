# -*- coding: utf-8 -*-
"""
Copyright 2020 huhui, Inc. All Rights Reserved.
@author: huhui
@software: PyCharm	
@project: yolo3SaftyHatDuJian	
@file: draw_anno.py	
@version: v1.0
@time: 2020/8/27 上午9:21
@setting: 
-------------------------------------------------
Description :
工程文件说明： 
"""

import shutil

# import cv2
# import os
#
# anno_path = 'train.txt'
# dst_dir = '/data/datasets/tianji/hat/anno'
# f = open(anno_path, 'r')
# for line in f.readlines():
#     contents = line.strip().split(' ')
#     img_path = contents[0]
#     img = cv2.imread(img_path)
#     for ann in contents[1:]:
#         x1, y1, x2, y2, label = list(map(int, ann.split(',')))
#         if label == 0:
#             color = (0, 255, 0)
#             text = 'hat'
#         else:
#             color = (0, 0, 255)
#             text = 'no hat'
#         img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
#         cv2.putText(img, text=text, org=(x1, y1 - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
#                     fontScale=0.50, color=color, thickness=2)
#     cv2.imwrite(os.path.join(dst_dir, os.path.basename(img_path)), img)
# f.close()

# p = '/data/datasets/tianji/hat/hat_val.txt'
# dst_dir = '/data/datasets/tianji/hat/val'
# f = open(p, 'r')
# for line in f.readlines():
#     img_path = line.split(' ')[0]
#     shutil.copy(img_path, dst_dir)
# f.close()


train_txt = '/data/datasets/tianji/hat/hat_train.txt'
pos_count = 0
neg_count = 0
f = open(train_txt, 'r')
for line in f.readlines():
    contents = line.strip().split(' ')
    img_path = contents[0]
    for ann in contents[1:]:
        x1, y1, x2, y2, label = list(map(int, ann.split(',')))
        if label == 0:
            pos_count += 1
        else:
            neg_count += 1
f.close()
print("hat count: {}, no hat count: {}".format(pos_count, neg_count))
