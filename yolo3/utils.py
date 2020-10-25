"""Miscellaneous utility functions."""

from functools import reduce
import pdb

from PIL import Image
import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

from yolo3.transform import MixupImage

def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')

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

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

<<<<<<< Updated upstream
<<<<<<< HEAD
def get_random_data(annotation_line, input_shape, random=True, max_boxes=200, jitter=.3, hue=.1, sat=1.5, val=1.5, proc_img=True):
=======
def get_random_data(annotation_line, input_shape, random=True, max_boxes=20, jitter=.3, hue=.1, sat=1.5, val=1.5, proc_img=True):
>>>>>>> e6598d13c703029b2686bc2eb8d5c09badf42992
=======
def get_mixup_image(image, box, bool_mixup=True, merge_line=None):
    iw, ih = image.size
    if bool_mixup:
        assert merge_line is not None
        merge_line = merge_line.split()
        merge_image = Image.open(merge_line[0])
        merge_iw, merge_ih = image.size
        merge_box = np.array([np.array(list(map(int,box.split(',')))) for box in merge_line[1:]])
        sample = {'image': np.array(image), 'gt_bbox': box[:,:4], 'gt_class': box[:, 4], 'gt_score': np.ones(len(box)),
                  'h': ih, 'w': iw,
                  'mixup': {'image': np.array(merge_image), 'gt_bbox': merge_box[:,:4], 'gt_class': merge_box[:, 4],
                            'gt_score': np.ones(len(merge_box)), 'h': merge_ih, 'w': merge_iw}}
        minup = MixupImage()
        out_sample = minup(sample)
    else:
        out_sample = {'image': np.array(image), 'gt_bbox': box[:,:4], 'gt_class': box[:, 4],
                      'gt_score': np.ones(len(box)), 'h': ih, 'w': iw}
    return out_sample


def get_random_data(annotation_line, input_shape, random=True, max_boxes=200, jitter=.3, hue=.1, sat=1.5, val=1.5,
                    proc_img=True, mixup=True, merge_line=None):
>>>>>>> Stashed changes
    '''random preprocessing for real-time data augmentation'''
    pdb.set_trace()
    line = annotation_line.split()
    image = Image.open(line[0])
    iw, ih = image.size
    h, w = input_shape
    box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

    if not random:
        # resize image
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)
        dx = (w-nw)//2
        dy = (h-nh)//2
        image_data=0
        if proc_img:
            image = image.resize((nw,nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image)/255.

        # correct boxes
        box_data = np.zeros((max_boxes,5))
        if len(box)>0:
            np.random.shuffle(box)
            if len(box)>max_boxes: box = box[:max_boxes]
            box[:, [0,2]] = box[:, [0,2]]*scale + dx
            box[:, [1,3]] = box[:, [1,3]]*scale + dy
            box_data[:len(box)] = box
            box_data = np.concatenate([box_data, np.ones(len(box))])

        return image_data, box_data

    sample = get_mixup_image(image, box, mixup, merge_line)
    image = Image.fromarray(sample['image'])
    box = np.concatenate((sample['gt_bbox'], np.expand_dims(sample['gt_class'], 1), np.expand_dims(sample['gt_score'], 1)), axis=1)
    iw, ih = sample['w'], sample['h']

    # resize image
    new_ar = w/h * rand(1-jitter,1+jitter)/rand(1-jitter,1+jitter)
    scale = rand(.25, 2)
    if new_ar < 1:
        nh = int(scale*h)
        nw = int(nh*new_ar)
    else:
        nw = int(scale*w)
        nh = int(nw/new_ar)
    image = image.resize((nw,nh), Image.BICUBIC)

    # place image
    dx = int(rand(0, w-nw))
    dy = int(rand(0, h-nh))
    new_image = Image.new('RGB', (w,h), (128,128,128))
    new_image.paste(image, (dx, dy))
    image = new_image

    # flip image or not
    flip = rand()<.5
    if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # distort image
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
    val = rand(1, val) if rand()<.5 else 1/rand(1, val)
    x = rgb_to_hsv(np.array(image)/255.)
    x[..., 0] += hue
    x[..., 0][x[..., 0]>1] -= 1
    x[..., 0][x[..., 0]<0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x>1] = 1
    x[x<0] = 0
    image_data = hsv_to_rgb(x) # numpy array, 0 to 1

    # correct boxes
    box_data = np.zeros((max_boxes,6))
    if len(box)>0:
        np.random.shuffle(box)
        box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
        box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
        if flip: box[:, [0,2]] = w - box[:, [2,0]]
        box[:, 0:2][box[:, 0:2]<0] = 0
        box[:, 2][box[:, 2]>w] = w
        box[:, 3][box[:, 3]>h] = h
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box
        if len(box)>max_boxes: box = box[:max_boxes]
        box_data[:len(box)] = box
    # import cv2
    # tmp = (image_data * 255).astype(np.uint8)
    # for s in box_data:
    #     tmp = cv2.rectangle(tmp, (int(s[0]), int(s[1])),
    #                         (int(s[2]), int(s[3])), (0, 0, 255), 2)
    # cv2.imwrite('tmp.jpg', tmp[:,:,::-1])

    return image_data, box_data
