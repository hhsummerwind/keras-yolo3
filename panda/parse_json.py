# -*- coding: utf-8 -*-
"""
Copyright 2021 huhui, Inc. All Rights Reserved.
@author: huhui
@software: PyCharm
@project: keras-yolo3	
@file: parse_json.py
@version: v1.0
@time: 2021/3/12 下午3:04
@setting: 
-------------------------------------------------
Description :
工程文件说明： 
"""

import os
import cv2
import sys
import json


def convert_json():
    json_path = '/data/datasets/PANDA/result/det_results_split3.json'
    result_path = '/data/datasets/PANDA/result/det_results_split3_right.json'
    f = open(json_path, 'r')
    result_list = []
    for line in f.readlines():
        if line == ']':
            break
        result_dict = line.strip().strip('[').strip(',')
        result_list.append(eval(result_dict))
    f.close()
    json.dump(result_list, open(result_path, 'w'))


def draw_result():
    json_path = '/data/datasets/PANDA/result/split4/result.json'#'/data/datasets/PANDA/result/det_results_split2_right.json'#'/data/datasets/PANDA/result/det_results_split1.json'
    map_json_path = '/data/datasets/PANDA/panda_round1_test_A_annos_202104/person_bbox_test_A.json'
    img_dir = '/data/datasets/PANDA/panda_round1_test_202104_A'

    classnames = ['visible_body', 'full_body', 'head', 'vehicle']
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 0)]  # blue green red black
    result_list = json.load(open(json_path, 'r'))
    map_imgid_path_dict = json.load(open(map_json_path, 'r'))
    map_dict = dict()
    for img_name, value in map_imgid_path_dict.items():
        current_img_id = value['image id']
        if current_img_id not in map_dict.keys():
            map_dict[current_img_id] = img_name

    last_image_id = None
    for result_dict in result_list:
        image_id = result_dict['image_id']
        if image_id != 500:
            continue
        # print(image_id)
        # sys.exit()
        if image_id != last_image_id:
            if last_image_id is not None:
                cv2.imwrite('tmp2.jpg', img)
                break
            last_image_id = image_id
            img_path = os.path.join(img_dir, map_dict[image_id])
            img = cv2.imread(img_path)
        category_id = result_dict['category_id']
        x1 = result_dict['bbox_left']
        y1 = result_dict['bbox_top']
        box_w = result_dict['bbox_width']
        box_h = result_dict['bbox_height']
        score = float(result_dict['score'])
        x2 = x1 + box_w
        y2 = y1 + box_h
        # if score > 0.3:
        img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), colors[category_id - 1], 6)
    cv2.imwrite('tmp2.jpg', img)


def merge_json():
    json_list = ['/data/datasets/PANDA/result/det_results_v1.json',
                 '/data/datasets/PANDA/result/det_results_split2_right.json',
                 '/data/datasets/PANDA/result/det_results_split3_right.json']
    result_path = '/data/datasets/PANDA/result/det_results123.json'
    result_list = []
    for json_path in json_list:
        result = json.load(open(json_path, 'r'))
        result_list.extend(result)
    json.dump(result_list, open(result_path, 'w'))


if __name__ == '__main__':
    draw_result()
    # convert_json()
    # merge_json()