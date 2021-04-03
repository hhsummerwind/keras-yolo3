# -*- coding: utf-8 -*-
"""
Copyright 2021 huhui, Inc. All Rights Reserved.
@author: huhui
@software: PyCharm
@project: keras-yolo3	
@file: parse_data.py
@version: v1.0
@time: 2021/3/7 下午2:33
@setting: 
-------------------------------------------------
Description :
工程文件说明： 
"""

import os
import json


def get_max_min(item, height, width, downsample_rate, valid_range, max_h, max_w, min_h, min_w):
    x1 = item['tl']['x']
    x2 = item['br']['x']
    y1 = item['tl']['y']
    y2 = item['br']['y']
    src_head_h = int((y2 - y1) * height)
    src_head_w = int((x2 - x1) * width)
    split2_head_h = int(src_head_h * downsample_rate)
    split2_head_w = int(src_head_w * downsample_rate)
    if split2_head_w > valid_range[1] or split2_head_h > valid_range[1] or split2_head_w < valid_range[0] or split2_head_h < valid_range[0]:
        return max_h, max_w, min_h, min_w
    if src_head_h > max_h:
        max_h = src_head_h
    if src_head_w > max_w:
        max_w = src_head_w
    if src_head_h < min_h:
        min_h = src_head_h
    if src_head_w < min_w:
        min_w = src_head_w
    return max_h, max_w, min_h, min_w


def check_box_in_range(item, height, width, downsample_rate):
    x1 = item['tl']['x']
    x2 = item['br']['x']
    y1 = item['tl']['y']
    y2 = item['br']['y']
    src_head_h = int((y2 - y1) * height)
    src_head_w = int((x2 - x1) * width)
    split2_head_h = int(src_head_h * downsample_rate)
    split2_head_w = int(src_head_w * downsample_rate)
    if split2_head_w > 400 or split2_head_h > 400 or split2_head_w < 1 or split2_head_h < 1:
        return False
    else:
        return True

def cal_box_range(obj_list, valid_range):
    img_dirs = ['/data/datasets/PANDA/panda_round1_train_202104_part1',
                '/data/datasets/PANDA/panda_round1_train_202104_part2']
    ann_dir = '/data/datasets/PANDA/panda_round1_train_annos_202104'
    person_ann_file = os.path.join(ann_dir, 'person_bbox_train.json')
    vehicle_ann_file = os.path.join(ann_dir, 'vehicle_bbox_train.json')
    person_ann = json.load(open(person_ann_file, 'r'))
    vehicle_ann = json.load(open(vehicle_ann_file, 'r'))
    max_h = 0
    min_h = 1e10
    max_w = 0
    min_w = 1e10
    for path, value in person_ann.items():
        image_id = value['image id']
        image_size = value['image size']
        height, width = image_size['height'], image_size['width']
        objects_list = value['objects list']
        downsample_rates = [1]#[416 / max(height, width), 1]
        for obj_dict in objects_list:
            category = obj_dict['category']
            if category != 'person':
                continue
            rects = obj_dict['rects']

            head = rects['head']
            visible_body = rects['visible body']
            full_body = rects['full body']

            for downsample_rate in downsample_rates:
                if 'head' in obj_list:
                    max_h, max_w, min_h, min_w = get_max_min(head, height, width, downsample_rate, valid_range, max_h, max_w, min_h, min_w)
                if 'visible_body' in obj_list:
                    max_h, max_w, min_h, min_w = get_max_min(visible_body, height, width, downsample_rate, valid_range, max_h, max_w, min_h, min_w)
                if 'full_body' in obj_list:
                    max_h, max_w, min_h, min_w = get_max_min(full_body, height, width, downsample_rate, valid_range, max_h, max_w, min_h, min_w)

        vehicle_value = vehicle_ann[path]
        vehicle_objects_list = vehicle_value['objects list']
        for obj_dict in vehicle_objects_list:
            category = obj_dict['category']
            if category == 'vehicles' or category == 'unsure':
                continue
            rect = obj_dict['rect']

            for downsample_rate in downsample_rates:
                if 'vehicle' in obj_list:
                    max_h, max_w, min_h, min_w = get_max_min(rect, height, width, downsample_rate, valid_range, max_h, max_w, min_h, min_w)

    # print(max_h, max_w, min_h, min_w)
    print("h: {}-{}, w: {}-{}".format(min_h, max_h, min_w, max_w))


def check_valid_box(item, height, width, downsample_rate, th=400):
    x1 = item['tl']['x']
    x2 = item['br']['x']
    y1 = item['tl']['y']
    y2 = item['br']['y']
    src_h = int((y2 - y1) * height)
    src_w = int((x2 - x1) * width)
    split2_h = int(src_h * downsample_rate)
    split2_w = int(src_w * downsample_rate)
    if split2_w > th or split2_h > th or split2_w < 1 or split2_h < 1:
        return False, src_h, src_w
    else:
        return True, src_h, src_w


def cal_num_obj():
    ann_dir = '/data/datasets/PANDA/panda_round1_train_annos_202104'
    person_ann_file = os.path.join(ann_dir, 'person_bbox_train.json')
    vehicle_ann_file = os.path.join(ann_dir, 'vehicle_bbox_train.json')
    person_ann = json.load(open(person_ann_file, 'r'))
    vehicle_ann = json.load(open(vehicle_ann_file, 'r'))
    max_num_obj = 0
    for path, value in person_ann.items():
        image_id = value['image id']
        image_size = value['image size']
        height, width = image_size['height'], image_size['width']
        objects_list = value['objects list']
        num_obj = 0
        downsample_rate = 416 / max(height, width)

        for obj_dict in objects_list:
            category = obj_dict['category']
            if category != 'person':
                continue
            rects = obj_dict['rects']

            head = rects['head']
            visible_body = rects['visible body']
            full_body = rects['full body']

            if check_valid_box(head, height, width, downsample_rate)[0]:
                num_obj += 1
            if check_valid_box(visible_body, height, width, downsample_rate)[0]:
                num_obj += 1
            if check_valid_box(full_body, height, width, downsample_rate)[0]:
                num_obj += 1

        vehicle_value = vehicle_ann[path]
        vehicle_objects_list = vehicle_value['objects list']
        for obj_dict in vehicle_objects_list:
            category = obj_dict['category']
            if category == 'vehicles' or category == 'unsure':
                continue
            rect = obj_dict['rect']

            if check_valid_box(rect, height, width, downsample_rate)[0]:
                num_obj += 1

        if num_obj > max_num_obj:
            max_num_obj = num_obj

    print(max_num_obj)


def cal_num_obj_crop():
    ann_txt = 'split3_annotations.txt'
    f = open(ann_txt, 'r')
    max_num_obj = 0

    for line in f.readlines():
        contents = line.strip().split('\t')
        num_obj = len(contents[1:])

        if num_obj > max_num_obj:
            max_num_obj = num_obj

    print(max_num_obj)


def get_invalid_box():
    img_dirs = ['/data/datasets/PANDA/panda_round1_train_202104_part1',
                '/data/datasets/PANDA/panda_round1_train_202104_part2']
    ann_dir = '/data/datasets/PANDA/panda_round1_train_annos_202104'
    person_ann_file = os.path.join(ann_dir, 'person_bbox_train.json')
    vehicle_ann_file = os.path.join(ann_dir, 'vehicle_bbox_train.json')
    person_ann = json.load(open(person_ann_file, 'r'))
    vehicle_ann = json.load(open(vehicle_ann_file, 'r'))
    for path, value in person_ann.items():
        image_id = value['image id']
        image_size = value['image size']
        height, width = image_size['height'], image_size['width']
        objects_list = value['objects list']
        downsample_rates = [608 / max(height, width), 1]
        ths = [600, 600]
        for obj_dict in objects_list:
            category = obj_dict['category']
            if category != 'person':
                continue
            rects = obj_dict['rects']

            head = rects['head']
            visible_body = rects['visible body']
            full_body = rects['full body']

            head_valid = False
            visible_body_valid = False
            full_body_valid = False
            for downsample_rate, th in zip(downsample_rates, ths):
                flag_head, src_head_h, src_head_w = check_valid_box(head, height, width, downsample_rate, th)
                if flag_head:
                    head_valid = True
                flag_visible_body, src_visible_body_h, src_visible_body_w = check_valid_box(visible_body, height, width, downsample_rate, th)
                if flag_visible_body:
                    visible_body_valid = True
                flag_full_body, src_full_body_h, src_full_body_w = check_valid_box(full_body, height, width, downsample_rate, th)
                if flag_full_body:
                    full_body_valid = True
            if not head_valid:
                print(path, src_head_h, src_head_w)
            if not visible_body_valid:
                print(path, src_visible_body_h, src_visible_body_w)
            if not full_body_valid:
                print(path, src_full_body_h, src_full_body_w)


        vehicle_value = vehicle_ann[path]
        vehicle_objects_list = vehicle_value['objects list']
        for obj_dict in vehicle_objects_list:
            category = obj_dict['category']
            if category == 'vehicles' or category == 'unsure':
                continue
            rect = obj_dict['rect']

            vehicle_valid = False
            for downsample_rate, th in zip(downsample_rates, ths):
                flag_vehicle, src_vehicle_h, src_vehicle_w = check_valid_box(rect, height, width, downsample_rate, th)
                if flag_vehicle:
                    vehicle_valid = True
            if not vehicle_valid:
                print(path, src_full_body_h, src_full_body_w)


if __name__ == '__main__':
    cal_num_obj_crop()
    # cal_box_range(['vehicle'], [0, 1e10])
    # get_invalid_box()