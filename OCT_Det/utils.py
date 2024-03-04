import json
import os

import numpy as np
from PIL import Image


def save_gif(img_dir, gif_path):
    images = []
    for filename in sorted(os.listdir(img_dir)):
        if filename.endswith('.png'):
            img_path = os.path.join(img_dir, filename)
            img = Image.open(img_path)
            images.append(img)

    images[0].save(gif_path, save_all=True, append_images=images[1:], duration=100, loop=0)


def resize_img(img_path, size):
    img = Image.open(img_path)
    if img.size == (size, size):
        return
    resized_img = img.resize((size, size), Image.ANTIALIAS)
    resized_img.save(img_path)


def conv2polygon(model, bbox_result, idx, score_thr):
    class_names = model.CLASSES
    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]
    ret = []
    for i, (bbox, label) in enumerate(zip(bboxes, labels)):
        bbox_int = bbox.astype(np.int32)
        poly = [[bbox_int[0], bbox_int[1]], [bbox_int[0], bbox_int[3]],
                [bbox_int[2], bbox_int[3]], [bbox_int[2], bbox_int[1]], idx]  # idx相当于Z维度的位置
        ret.append({"label": class_names[label], "poly": poly, "score": bbox[-1]})
    return ret


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
