# -*- coding: utf-8 -*-
"""
Copyright 2020 huhui, Inc. All Rights Reserved.
@author: huhui
@software: PyCharm	
@project: keras-yolo3	
@file: eval.py	
@version: v1.0
@time: 2020/8/8 下午8:46
@setting: 
-------------------------------------------------
Description :
工程文件说明： 
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from collections import defaultdict
from PIL import Image
from keras import backend as K
from PIL import Image, ImageFont, ImageDraw
from timeit import default_timer as timer

from yolo import YOLO


VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')
Color = [[0, 0, 0],
         [128, 0, 0],
         [0, 128, 0],
         [128, 128, 0],
         [0, 0, 128],
         [128, 0, 128],
         [0, 128, 128],
         [128, 128, 128],
         [64, 0, 0],
         [192, 0, 0],
         [64, 128, 0],
         [192, 128, 0],
         [64, 0, 128],
         [192, 0, 128],
         [64, 128, 128],
         [192, 128, 128],
         [0, 64, 0],
         [128, 64, 0],
         [0, 192, 0],
         [128, 192, 0],
         [0, 64, 128]]


def voc_ap(rec, prec, use_07_metric=False):
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.

    else:
        # correct ap caculation
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        i = np.where(mrec[1:] != mrec[:-1])[0]

        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap


def voc_eval(preds, target, VOC_CLASSES=VOC_CLASSES, threshold=0.5, use_07_metric=False, ):
    '''
    preds {'cat':[[image_id,confidence,x1,y1,x2,y2],...],'dog':[[],...]}
    target {(image_id,class):[[],]}
    '''
    aps = []
    for i, class_ in enumerate(VOC_CLASSES):
        pred = preds[class_]  # [[image_id,confidence,x1,y1,x2,y2],...]
        if len(pred) == 0:  # 如果这个类别一个都没有检测到的异常情况
            ap = -1
            print('---class {} ap {}---'.format(class_, ap))
            aps += [ap]
            break
        # print(pred)
        image_ids = [x[0] for x in pred]
        confidence = np.array([float(x[1]) for x in pred])
        BB = np.array([x[2:] for x in pred])
        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        npos = 0.
        for (key1, key2) in target:
            if key2 == class_:
                npos += len(target[(key1, key2)])  # 统计这个类别的正样本，在这里统计才不会遗漏
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d, image_id in enumerate(image_ids):
            bb = BB[d]  # 预测框
            if (image_id, class_) in target:
                BBGT = target[(image_id, class_)]  # [[],]
                for bbgt in BBGT:
                    # compute overlaps
                    # intersection
                    ixmin = np.maximum(bbgt[0], bb[0])
                    iymin = np.maximum(bbgt[1], bb[1])
                    ixmax = np.minimum(bbgt[2], bb[2])
                    iymax = np.minimum(bbgt[3], bb[3])
                    iw = np.maximum(ixmax - ixmin + 1., 0.)
                    ih = np.maximum(iymax - iymin + 1., 0.)
                    inters = iw * ih

                    union = (bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) + (bbgt[2] - bbgt[0] + 1.) * (
                            bbgt[3] - bbgt[1] + 1.) - inters
                    if union == 0:
                        print(bb, bbgt)

                    overlaps = inters / union
                    if overlaps > threshold:
                        tp[d] = 1
                        BBGT.remove(bbgt)  # 这个框已经匹配到了，不能再匹配
                        if len(BBGT) == 0:
                            del target[(image_id, class_)]  # 删除没有box的键值
                        break
                fp[d] = 1 - tp[d]
            else:
                fp[d] = 1
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        # print(rec,prec)
        ap = voc_ap(rec, prec, use_07_metric)
        print('---class {} ap {}---'.format(class_, ap))
        aps += [ap]
    print('---map {}---'.format(np.mean(aps)))

def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image

def detect_image(yolo, image, img_id, preds, view=True):
    start = timer()

    if yolo.model_image_size != (None, None):
        assert yolo.model_image_size[0]%32 == 0, 'Multiples of 32 required'
        assert yolo.model_image_size[1]%32 == 0, 'Multiples of 32 required'
        boxed_image = letterbox_image(image, tuple(reversed(yolo.model_image_size)))
    else:
        new_image_size = (image.width - (image.width % 32),
                          image.height - (image.height % 32))
        boxed_image = letterbox_image(image, new_image_size)
    image_data = np.array(boxed_image, dtype='float32')

    print(image_data.shape)
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

    out_boxes, out_scores, out_classes = yolo.sess.run(
        [yolo.boxes, yolo.scores, yolo.classes],
        feed_dict={
            yolo.yolo_model.input: image_data,
            yolo.input_image_shape: [image.size[1], image.size[0]],
            K.learning_phase(): 0
        })

    print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

    if view:
        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = yolo.class_names[c]
        box = out_boxes[i]
        score = out_scores[i]

        label = '{} {:.2f}'.format(predicted_class, score)
        if view:
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        print(label, (left, top), (right, bottom))

        preds[predicted_class].append([img_id, score, left, top, right, bottom])

        if view:
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=yolo.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=yolo.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

    end = timer()
    print(end - start)
    return image

def eval():
    test_txt = 'voc/2007_test.txt'
    yolo = YOLO()
    f = open(test_txt, 'r')
    target = defaultdict(list)
    preds = defaultdict(list)
    for line in f.readlines():
        contents = line.strip().split(' ')
        img_path = contents[0]
        img_id = os.path.basename(img_path)
        for label in contents[1:]:
            x1, y1, x2, y2, class_id = list(map(int, label.split(',')))
            target[(img_id, VOC_CLASSES[class_id])].append([x1, y1, x2, y2])
        image = Image.open(img_path)
        image = detect_image(yolo, image, img_id, preds, view=False)
    f.close()
    print('----------------------------------')
    print(preds)
    print('----------------------------------')
    print(target)
    print('----------------------------------')
    voc_eval(preds, target, VOC_CLASSES=VOC_CLASSES)


if __name__ == '__main__':
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
    sess = tf.Session(config=config)
    set_session(tf.Session(config=config))
    eval()