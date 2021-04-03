# -*- coding: utf-8 -*-
"""
Copyright 2021 huhui, Inc. All Rights Reserved.
@author: huhui
@software: PyCharm
@project: keras-yolo3	
@file: split_image.py
@version: v1.0
@time: 2021/3/3 下午4:32
@setting: 
-------------------------------------------------
Description :
工程文件说明： 
"""

import os
import cv2
import json
import numpy as np

img_dirs = ['/data/datasets/PANDA/panda_round1_train_202104_part1',
            '/data/datasets/PANDA/panda_round1_train_202104_part2']
ann_dir = '/data/datasets/PANDA/panda_round1_train_annos_202104'
person_ann_file = os.path.join(ann_dir, 'person_bbox_train.json')
vehicle_ann_file = os.path.join(ann_dir, 'vehicle_bbox_train.json')
person_ann = json.load(open(person_ann_file, 'r'))
vehicle_ann = json.load(open(vehicle_ann_file, 'r'))


def get_split1():
    out_dir = '/data/datasets/PANDA/split1'
    box_th = 1
    target = 800
    out_ann_file = 'split1_annotations.txt'
    f = open(out_ann_file, 'w')
    for path, value in person_ann.items():
        print(path)
        if int(path.split('_')[0]) <= 8:
            img_dir = img_dirs[0]
        else:
            img_dir = img_dirs[1]
        img_path = os.path.join(img_dir, path)
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        if h >= w:
            rate = h / target
            target_shape = (int(w / rate), target)
        else:
            rate = w / target
            target_shape = (target, int(h / rate))
        img = cv2.resize(img, target_shape)
        dst_path = os.path.join(out_dir, os.path.basename(path))
        cv2.imwrite(dst_path, img)
        f.write(dst_path + '\t')

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
            head_x1, head_y1, head_x2, head_y2 = \
                float(head['tl']['x']) * width, \
                float(head['tl']['y']) * height, \
                float(head['br']['x']) * width, \
                float(head['br']['y']) * height,
            visible_body_x1, visible_body_y1, visible_body_x2, visible_body_y2 = \
                float(visible_body['tl']['x']) * width, \
                float(visible_body['tl']['y']) * height, \
                float(visible_body['br']['x']) * width, \
                float(visible_body['br']['y']) * height,
            full_body_x1, full_body_y1, full_body_x2, full_body_y2 = \
                float(full_body['tl']['x']) * width, \
                float(full_body['tl']['y']) * height, \
                float(full_body['br']['x']) * width, \
                float(full_body['br']['y']) * height,
            if (head_x2 - head_x1) / rate > box_th and (head_y2 - head_y1) / rate > box_th:
                f.write('{},{},{},{},2\t'.format(int(head_x1 / width * target_shape[0]),
                                                 int(head_y1 / height * target_shape[1]),
                                                 int(head_x2 / width * target_shape[0]),
                                                 int(head_y2 / height * target_shape[1])))
            if (visible_body_x2 - visible_body_x1) / rate > box_th and (
                    visible_body_y2 - visible_body_y1) / rate > box_th:
                f.write('{},{},{},{},0\t'.format(int(visible_body_x1 / width * target_shape[0]),
                                                 int(visible_body_y1 / height * target_shape[1]),
                                                 int(visible_body_x2 / width * target_shape[0]),
                                                 int(visible_body_y2 / height * target_shape[1])))
            if (full_body_x2 - full_body_x1) / rate > box_th and (full_body_y2 - full_body_y1) / rate > box_th:
                f.write('{},{},{},{},1\t'.format(int(full_body_x1 / width * target_shape[0]),
                                                 int(full_body_y1 / height * target_shape[1]),
                                                 int(full_body_x2 / width * target_shape[0]),
                                                 int(full_body_y2 / height * target_shape[1])))

        vehicle_value = vehicle_ann[path]
        vehicle_objects_list = vehicle_value['objects list']
        for obj_dict in vehicle_objects_list:
            category = obj_dict['category']
            if category == 'vehicles' or category == 'unsure':
                continue
            rect = obj_dict['rect']
            vehicle_x1, vehicle_y1, vehicle_x2, vehicle_y2 = \
                float(rect['tl']['x']) * width, \
                float(rect['tl']['y']) * height, \
                float(rect['br']['x']) * width, \
                float(rect['br']['y']) * height,
            if (vehicle_x2 - vehicle_x1) / rate > box_th and (vehicle_y2 - vehicle_y1) / rate > box_th:
                f.write('{},{},{},{},3\t'.format(int(vehicle_x1 / width * target_shape[0]),
                                                 int(vehicle_y1 / height * target_shape[1]),
                                                 int(vehicle_x2 / width * target_shape[0]),
                                                 int(vehicle_y2 / height * target_shape[1])))
        f.write('\n')
    f.close()


def cal_area(box):
    x1, y1, x2, y2 = box
    return (x2 - x1) * (y2 - y1)


def cal_intersection(box1, box2):
    x11, y11, x12, y12 = box1
    x21, y21, x22, y22 = box2
    intersection = (max(x21, x11), max(y21, y11), min(x12, x22), min(y22, y12))
    if intersection[0] < intersection[2] and intersection[1] < intersection[3]:
        return intersection
    else:
        return -1


def cal_iou(box1, box2):
    area1 = cal_area(box1)
    area2 = cal_area(box2)
    intersection = cal_intersection(box1, box2)
    if intersection == -1:
        return 0
    intersection_area = cal_area(intersection)
    union_area = area1 + area2 - intersection_area
    iou = intersection_area / union_area
    return iou


def get_split_downsample(out_dir, downsample_rate, out_ann_file, box_th=400, target=416):
    # out_dir = '/data/datasets/PANDA/split3'
    # box_th = 400
    # target = 416
    # downsample_rate = 0.5
    # out_ann_file = 'split3_annotations.txt'
    f = open(out_ann_file, 'w')
    for path, value in person_ann.items():
        # if path != '01_University_Canteen/IMG_01_02.jpg':
        #     continue
        print(path)
        if int(path.split('_')[0]) <= 8:
            img_dir = img_dirs[0]
        else:
            img_dir = img_dirs[1]

        image_size = value['image size']
        height, width = image_size['height'], image_size['width']
        height = int(height * downsample_rate)
        width = int(width * downsample_rate)
        objects_list = value['objects list']
        label_list = []
        rect_list = []
        for obj_dict in objects_list:
            category = obj_dict['category']
            if category != 'person':
                continue
            rects = obj_dict['rects']
            head = rects['head']
            visible_body = rects['visible body']
            full_body = rects['full body']
            head_x1, head_y1, head_x2, head_y2 = \
                int(float(head['tl']['x']) * width), \
                int(float(head['tl']['y']) * height), \
                int(float(head['br']['x']) * width), \
                int(float(head['br']['y']) * height)
            visible_body_x1, visible_body_y1, visible_body_x2, visible_body_y2 = \
                int(float(visible_body['tl']['x']) * width), \
                int(float(visible_body['tl']['y']) * height), \
                int(float(visible_body['br']['x']) * width), \
                int(float(visible_body['br']['y']) * height)
            full_body_x1, full_body_y1, full_body_x2, full_body_y2 = \
                int(float(full_body['tl']['x']) * width), \
                int(float(full_body['tl']['y']) * height), \
                int(float(full_body['br']['x']) * width), \
                int(float(full_body['br']['y']) * height)

            if (full_body_x2 - full_body_x1) < box_th or (full_body_y2 - full_body_y1) < box_th:
                label_list.append(1)
                rect_list.append([full_body_x1, full_body_y1, full_body_x2, full_body_y2])
                label_list.append(0)
                rect_list.append([visible_body_x1, visible_body_y1, visible_body_x2, visible_body_y2])
                label_list.append(2)
                rect_list.append([head_x1, head_y1, head_x2, head_y2])
            elif (visible_body_x2 - visible_body_x1) < box_th or (visible_body_y2 - visible_body_y1) < box_th:
                label_list.append(0)
                rect_list.append([visible_body_x1, visible_body_y1, visible_body_x2, visible_body_y2])
                label_list.append(2)
                rect_list.append([head_x1, head_y1, head_x2, head_y2])
            elif (head_x2 - head_x1) < box_th or (head_y2 - head_y1) < box_th:
                label_list.append(2)
                rect_list.append([head_x1, head_y1, head_x2, head_y2])

        vehicle_value = vehicle_ann[path]
        vehicle_objects_list = vehicle_value['objects list']
        for obj_dict in vehicle_objects_list:
            category = obj_dict['category']
            if category == 'vehicles' or category == 'unsure':
                continue
            rect = obj_dict['rect']
            vehicle_x1, vehicle_y1, vehicle_x2, vehicle_y2 = \
                int(float(rect['tl']['x']) * width), \
                int(float(rect['tl']['y']) * height), \
                int(float(rect['br']['x']) * width), \
                int(float(rect['br']['y']) * height)
            if (vehicle_x2 - vehicle_x1) < box_th or (vehicle_y2 - vehicle_y1) < box_th:
                label_list.append(3)
                rect_list.append([vehicle_x1, vehicle_y1, vehicle_x2, vehicle_y2])

        img_path = os.path.join(img_dir, path)
        src_img = cv2.imread(img_path)
        src_img = cv2.resize(src_img, (width, height))
        for i, (label, rect) in enumerate(zip(label_list, rect_list)):
            if downsample_rate >= 0.1 and label == 1:
                continue
            elif downsample_rate < 0.1:
                if label == 0 or label == 2:
                    continue
            # if label != 2:
            #     continue
            # print('\t', i)
            target_img = np.ones((target, target, 3), dtype=np.uint8) * 127
            x1, y1, x2, y2 = rect
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            x_start = int(center_x - target / 2)
            x_end = int(x_start + target)
            y_start = int(center_y - target / 2)
            y_end = int(y_start + target)
            cropped_img = src_img[max(y_start, 0):min(y_end, height), max(x_start, 0):min(x_end, width)]
            if len(cropped_img) == 0:
                continue
            target_img[
                max(0, -y_start):min(target, target - (y_end - height)),
                max(0, -x_start):min(target, target - (x_end - width))
            ] = cropped_img
            dst_path = os.path.join(out_dir, '{}_{}.jpg'.format(os.path.splitext(os.path.basename(img_path))[0], i))
            cv2.imwrite(dst_path, target_img)
            f.write(dst_path + '\t')
            for j, (label_in_range, rect_in_range) in enumerate(zip(label_list, rect_list)):
                # if label_in_range != 2:
                #     continue
                if label_in_range == 1:
                    target_x1 = rect_in_range[0] - x_start
                    target_y1 = rect_in_range[1] - y_start
                    target_x2 = rect_in_range[2] - x_start
                    target_y2 = rect_in_range[3] - y_start
                else:
                    target_x1 = max(0, rect_in_range[0] - x_start)
                    target_y1 = max(0, rect_in_range[1] - y_start)
                    target_x2 = min(target, rect_in_range[2] - x_start)
                    target_y2 = min(target, rect_in_range[3] - y_start)
                    if target_x2 <= target_x1 or target_y2 <= target_y1:
                        continue
                intersection = cal_intersection(rect_in_range, (x_start, y_start, x_end, y_end))
                if intersection == -1:
                    continue
                if label_in_range == 1 and cal_iou(intersection, rect_list[j + 1]) == 0:
                    continue
                iou_target = cal_iou(rect_in_range, intersection)
                if label_in_range == 1 and iou_target > 1/10:
                    f.write('{},{},{},{},{}\t'.format(target_x1, target_y1, target_x2, target_y2, label_in_range))
                elif label_in_range != 1 and iou_target > 1/10:
                    f.write('{},{},{},{},{}\t'.format(target_x1, target_y1, target_x2, target_y2, label_in_range))
                # f.write('{},{},{},{},{}\t'.format(target_x1, target_y1, target_x2, target_y2, label_in_range))
            f.write('\n')
    f.close()


def get_split2():
    out_dir = '/data/datasets/PANDA/split2'
    box_th = 400
    target = 416
    out_ann_file = 'split2_annotations.txt'
    downsample_rate = 1
    get_split_downsample(out_dir, downsample_rate, out_ann_file, box_th, target)


def get_split2_608():
    out_dir = '/data/datasets/PANDA/split2_608'
    box_th = 600
    target = 608
    out_ann_file = 'split2_608_annotations.txt'
    downsample_rate = 1
    get_split_downsample(out_dir, downsample_rate, out_ann_file, box_th, target)


def get_split3():
    out_dir = '/data/datasets/PANDA/split3'
    box_th = 400
    target = 416
    out_ann_file = 'split3_annotations.txt'
    downsample_rate = 0.5
    get_split_downsample(out_dir, downsample_rate, out_ann_file, box_th, target)


def get_split4():
    out_dir = '/data/datasets/PANDA/split4'
    box_th = 400
    target = 416
    out_ann_file = 'split4_annotations.txt'
    downsample_rate = 0.05
    get_split_downsample(out_dir, downsample_rate, out_ann_file, box_th, target)


if __name__ == '__main__':
    # get_split1()
    # get_split2()
    # get_split3()
    # get_split4()
    get_split2_608()