# -*- coding: utf-8 -*-
"""
Copyright 2021 huhui, Inc. All Rights Reserved.
@author: huhui
@software: PyCharm
@project: keras-yolo3	
@file: test.py
@version: v1.0
@time: 2021/3/8 下午10:59
@setting: 
-------------------------------------------------
Description :
工程文件说明： 
"""

import colorsys
import json
from timeit import default_timer as timer
from tqdm import tqdm

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import sys
import pdb

sys.path.append('/projects/open_sources/object_detection/keras-yolo3')
# sys.path.append('/home/huhui/Documents/open_sources/object_detection/keras-yolo3')
from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model


class YOLO(object):
    _defaults = {
        "model_path": '/data/models/panda/yolov3_split1/trained_weights_final.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'panda/class_names.txt',
        "score": 0.3,
        "iou": 0.45,
        "model_image_size": (416, 416),
        "gpu_num": 1,
        "max_boxes": 2000
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)  # set up default values
        self.__dict__.update(kwargs)  # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors == 6  # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None, None, 3)), num_anchors // 2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None, None, 3)), num_anchors // 3, num_classes)
            self.yolo_model.load_weights(self.model_path)  # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                   num_anchors / len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2,))
        if self.gpu_num >= 2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                                           len(self.class_names), self.input_image_shape,
                                           score_threshold=self.score, iou_threshold=self.iou,
                                           max_boxes=self.max_boxes)
        return boxes, scores, classes

    def detect_image(self, image, save_path=None, use_text=False):
        start = timer()
        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        # print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
        # image_data = np.stack(input_imgs)

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: self.model_image_size,
                K.learning_phase(): 0
            })

        # print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        #
        # if save_path is not None:
        #     font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
        #                               size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        #     thickness = max((image.size[0] + image.size[1]) // 600, 1)

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            # print(label)
            # if save_path is not None:
            #     draw = ImageDraw.Draw(image)
            #     label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            # print(label, (left, top), (right, bottom))
            yield c + 1, score, left, top, right, bottom

            # if save_path is not None:
            #     if top - label_size[1] >= 0:
            #         text_origin = np.array([left, top - label_size[1]])
            #     else:
            #         text_origin = np.array([left, top + 1])
            #
            #     # My kingdom for a good redistributable image drawing library.
            #     for i in range(thickness):
            #         draw.rectangle(
            #             [left + i, top + i, right - i, bottom - i],
            #             outline=self.colors[c])
            #     if use_text:
            #         draw.rectangle(
            #             [tuple(text_origin), tuple(text_origin + label_size)],
            #             fill=self.colors[c])
            #         draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            # if save_path is not None:
            #     del draw

        end = timer()
        # print(end - start)
        # return image
        # if save_path is not None:
        #     image.save(save_path)

    def close_session(self):
        self.sess.close()


def get_cropped_imgs(src_img, crop_size, overlap, padding_size, downsample_rate=1):
    assert overlap < crop_size
    h, w = src_img.shape[:2]
    assert padding_size < h
    assert padding_size < w
    padding_shape = (h + 2 * padding_size, w + 2 * padding_size, 3)
    padding_img = np.ones(padding_shape, dtype=np.uint8) * 127
    padding_img[padding_size: padding_size + h, padding_size:padding_size + w, :] = src_img
    h_start_list = []
    h_end_list = []
    w_start_list = []
    w_end_list = []

    h_start = 0
    while True:
        h_end = h_start + crop_size
        if h_end <= padding_size:
            continue
        else:
            h_start_list.append(h_start)
        if h_end >= padding_shape[0]:
            h_end_list.append(padding_shape[0])
            break
        else:
            h_end_list.append(h_end)
            if h_end >= padding_shape[0] - overlap:
                break
        h_start = h_end - overlap

    w_start = 0
    while True:
        w_end = w_start + crop_size
        if w_end <= padding_size:
            continue
        else:
            w_start_list.append(w_start)
        if w_end >= padding_shape[1]:
            w_end_list.append(padding_shape[1])
            break
        else:
            w_end_list.append(w_end)
            if w_end >= padding_shape[1] - overlap:
                break
        w_start = w_end - overlap

    assert len(h_start_list) == len(h_end_list)
    assert len(w_start_list) == len(w_end_list)

    h_range_list = list(zip(h_start_list, h_end_list))  # 原图的尺寸起始值
    w_range_list = list(zip(w_start_list, w_end_list))

    import itertools
    for i, item in enumerate(itertools.product(h_range_list, w_range_list)):
        h_range = item[0]
        w_range = item[1]
        result_img = np.ones((crop_size, crop_size, 3), dtype=np.uint8) * 127
        result_h_start = int((crop_size - (h_range[1] - h_range[0])) / 2)  # 若截出的图小于crop_size，则在截出的图周边补全，在输出图中的起始位置
        result_w_start = int((crop_size - (w_range[1] - w_range[0])) / 2)
        result_img[result_h_start:result_h_start + h_range[1] - h_range[0],
        result_w_start:result_w_start + w_range[1] - w_range[0]] = padding_img[h_range[0]:h_range[1],
                                                                   w_range[0]:w_range[1]]

        def map_x1(x):
            x1 = max(x + w_range[0] - padding_size, 0)
            if x1 >= w:
                return None
            else:
                return int(x1 / downsample_rate)

        def map_y1(y):
            y1 = max(y + h_range[0] - padding_size, 0)
            if y1 >= h:
                return None
            else:
                return int(y1 / downsample_rate)

        def map_x2(x):
            x2 = min(x + w_range[0] - padding_size, w)
            if x2 <= 0:
                return None
            else:
                return int(x2 / downsample_rate)

        def map_y2(y):
            y2 = min(y + h_range[0] - padding_size, h)
            if y2 <= 0:
                return None
            else:
                return int(y2 / downsample_rate)

        yield result_img, map_x1, map_y1, map_x2, map_y2, i == len(h_range_list) * len(w_range_list) - 1


def test():
    """
        "image_id":1,
		"category_id":1,
		"bbox_left":500,
		"bbox_top":500,
		"bbox_width":500,
		"bbox_height":500,
		"score":0.8
    """
    data_dir = '/data/datasets/PANDA'
    out_path = '/data/datasets/PANDA/result/det_results_split1.json'
    target_size = 416
    ann_dir = os.path.join(data_dir, 'panda_round1_test_A_annos_202104')
    img_dir = os.path.join(data_dir, 'panda_round1_test_202104_A')
    person_ann_path = os.path.join(ann_dir, 'person_bbox_test_A.json')
    vechicle_ann_path = os.path.join(ann_dir, 'vehicle_bbox_test_A.json')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
    sess = tf.Session(config=config)
    set_session(sess)

    result_list = []
    person_ann = json.load(open(person_ann_path, 'r'))
    yolo = YOLO()
    for path, info in person_ann.items():
        print(path)
        img_path = os.path.join(img_dir, path)
        img_id = info['image id']
        h, w = info['image size']['height'], info['image size']['width']
        image = Image.open(img_path)
        for label, score, left, top, right, bottom in yolo.detect_image([image]):
            result_list.append({'image_id': img_id, 'category_id': int(label), 'bbox_left': int(left),
                                'bbox_top': int(top), 'bbox_width': int(right - left), 'bbox_height': int(bottom - top),
                                'score': float(score)})
        # break
    json.dump(result_list, open(out_path, 'w'))


def test_crop():
    import cv2
    data_dir = '/data/datasets/PANDA'
    out_path = '/data/datasets/PANDA/result/det_results_split2.json'
    params = {
        "model_path": '/data/models/panda/yolov3_split2/ep048-loss87.895-val_loss87.852.h5',
        "anchors_path": 'panda/split2_anchors.txt',
        "classes_path": 'panda/class_names.txt',
        "score": 0.55,
        "iou": 0.45,
        "model_image_size": (416, 416),
        "gpu_num": 1,
        "max_boxes": 200
    }

    crop_size = 416
    overlap = 100
    padding_size = 0
    batch_size = 20
    ann_dir = os.path.join(data_dir, 'panda_round1_test_A_annos_202104')
    img_dir = os.path.join(data_dir, 'panda_round1_test_202104_A')
    person_ann_path = os.path.join(ann_dir, 'person_bbox_test_A.json')
    vechicle_ann_path = os.path.join(ann_dir, 'vehicle_bbox_test_A.json')

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
    sess = tf.Session(config=config)
    set_session(sess)

    person_ann = json.load(open(person_ann_path, 'r'))
    yolo = YOLO(**params)
    f = open(out_path, 'w')
    f.write('[')
    for i, (path, info) in enumerate(person_ann.items()):
        print(path)
        img_path = os.path.join(img_dir, path)
        img_id = info['image id']
        h, w = info['image size']['height'], info['image size']['width']
        image = cv2.imread(img_path)[..., ::-1]
        # image = np.ascontiguousarray(image, dtype=np.uint8)

        for cropped_img, map_x1, map_y1, map_x2, map_y2, flag in tqdm(
                get_cropped_imgs(image, crop_size, overlap, padding_size)):
            cropped_pil_img = Image.fromarray(cropped_img)
            results = yolo.detect_image(cropped_pil_img)
            for label, score, left, top, right, bottom in results:

                x1, y1, x2, y2 = map_x1(left), map_y1(top), map_x2(right), map_y2(bottom)
                if x1 is None or x2 is None or y1 is None or y2 is None:
                    continue

                # image = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 3)

                result = {'image_id': img_id, 'category_id': int(label), 'bbox_left': int(x1),
                          'bbox_top': int(y1), 'bbox_width': int(x2 - x1),
                          'bbox_height': int(y2 - y1),
                          'score': float(score)}
                json.dump(result, f)
                f.write(',\n')
        # cv2.imwrite('tmp.jpg', image[..., ::-1])
        # break
    f.write(']')
    f.close()

    # for label, score, left, top, right, bottom in yolo.detect_image(image):
    #     result_list.append({'image_id': img_id, 'category_id': int(label), 'bbox_left': int(left),
    #                         'bbox_top': int(top), 'bbox_width': int(right - left), 'bbox_height': int(bottom - top),
    #                         'score': float(score)})
    # break
    # json.dump(result_list, open(out_path, 'w'))


def cp():
    import glob
    import shutil
    src_dir = '/data/datasets/PANDA/split3'
    dst_dir = '/data/datasets/PANDA/tmp3'
    for i, img_path in enumerate(glob.glob(os.path.join(src_dir, '*'))):
        # if i < 10:
        #     continue
        if i == 50:
            break
        shutil.copy(img_path, dst_dir)


def test_crop_split3():
    import cv2
    data_dir = '/data/datasets/PANDA'
    out_path = '/data/datasets/PANDA/result/det_results_split3.json'
    params = {
        "model_path": '/data/models/panda/yolov3_split3/trained_weights_final.h5',
        "anchors_path": 'panda/split3_anchors.txt',
        "classes_path": 'panda/class_names.txt',
        "score": 0.55,
        "iou": 0.45,
        "model_image_size": (416, 416),
        "gpu_num": 1,
        "max_boxes": 300
    }

    crop_size = 416
    overlap = 100
    padding_size = 0
    downsample_rate = 0.5
    ann_dir = os.path.join(data_dir, 'panda_round1_test_A_annos_202104')
    img_dir = os.path.join(data_dir, 'panda_round1_test_202104_A')
    person_ann_path = os.path.join(ann_dir, 'person_bbox_test_A.json')
    vechicle_ann_path = os.path.join(ann_dir, 'vehicle_bbox_test_A.json')

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
    sess = tf.Session(config=config)
    set_session(sess)

    person_ann = json.load(open(person_ann_path, 'r'))
    yolo = YOLO(**params)
    f = open(out_path, 'w')
    f.write('[')
    for i, (path, info) in enumerate(person_ann.items()):
        print(path)
        img_path = os.path.join(img_dir, path)
        img_id = info['image id']
        h, w = info['image size']['height'], info['image size']['width']
        image = cv2.imread(img_path)[..., ::-1]
        target_size = (int(w / 2), int(h / 2))
        downsample_image = cv2.resize(image, target_size)
        # image = np.ascontiguousarray(image, dtype=np.uint8)

        for cropped_img, map_x1, map_y1, map_x2, map_y2, flag in tqdm(
                get_cropped_imgs(downsample_image, crop_size, overlap, padding_size, downsample_rate)):
            cropped_pil_img = Image.fromarray(cropped_img)
            results = yolo.detect_image(cropped_pil_img)
            for label, score, left, top, right, bottom in results:

                x1, y1, x2, y2 = map_x1(left), map_y1(top), map_x2(right), map_y2(bottom)
                if x1 is None or x2 is None or y1 is None or y2 is None:
                    continue

                # pdb.set_trace()
                # image = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 3)

                result = {'image_id': img_id, 'category_id': int(label), 'bbox_left': int(x1),
                          'bbox_top': int(y1), 'bbox_width': int(x2 - x1),
                          'bbox_height': int(y2 - y1),
                          'score': float(score)}
                json.dump(result, f)
                f.write(',\n')
        # cv2.imwrite('tmp.jpg', cv2.resize(image[..., ::-1], target_size))
        # break
    f.write(']')
    f.close()

    # for label, score, left, top, right, bottom in yolo.detect_image(image):
    #     result_list.append({'image_id': img_id, 'category_id': int(label), 'bbox_left': int(left),
    #                         'bbox_top': int(top), 'bbox_width': int(right - left), 'bbox_height': int(bottom - top),
    #                         'score': float(score)})
    # break
    # json.dump(result_list, open(out_path, 'w'))


if __name__ == '__main__':
    # test()
    # cp()
    # test_crop()
    test_crop_split3()
