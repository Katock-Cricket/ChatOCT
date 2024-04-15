import json
import os
import re

import imageio.v2 as imageio
import numpy as np
from PIL import Image


def sort_by_frame(img_list):
    frame_regex = r'frame(\d+)'

    def frame_key(path):
        match = re.search(frame_regex, path)
        if match:
            return int(match.group(1))
        else:
            return 0

    sorted_img_list = sorted(img_list, key=frame_key)
    return sorted_img_list


def save_gif(img_dir, gif_path):
    gif_frames = []
    img_list = sort_by_frame(os.listdir(img_dir))
    for filename in img_list:
        if filename.endswith('.png'):
            img_path = os.path.join(img_dir, filename)
            gif_frames.append(imageio.imread(img_path))

    imageio.mimsave(gif_path, gif_frames, 'GIF', duration=50)


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
