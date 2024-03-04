import json
import os.path

import numpy as np
from sklearn.cluster import DBSCAN
from argparse import ArgumentParser


class OCTClass:
    def __init__(self):
        self.num = 0
        self.obj_list = []
        self.class_name = ''

    def add_obj(self, obj):
        self.obj_list.append(obj)
        self.num += 1

    def __str__(self):
        ret = f"检测到疑似{self.class_name}{self.num}处\n"
        for idx, obj in enumerate(self.obj_list):
            rate = obj.max_square / actual_s
            ret += f"第{idx + 1}处，"
            ret += f"置信度{obj.score * 100:.2f}%，"
            ret += f"最大截面积约占血管{rate * 100:.2f}%，"
            ret += f"长度约{obj.length:.2f}毫米，"
            ret += f"体积约{obj.volume:.2f}立方毫米\n"
        return ret


class JS(OCTClass):
    class Cluster:
        def __init__(self):
            self.js_num = 0
            self.js_list = []
            self.score = 0

        def add_js(self, js):
            self.js_num += 1
            self.js_list.append(js)
            self.score = (self.score * (self.js_num - 1) + js[3]) / self.js_num

        def __str__(self):
            ret = f"含有巨噬细胞{self.js_num}个，"
            ret += f"置信度{self.score * 100:.2f}%\n"
            return ret

    def __init__(self):
        super().__init__()
        self.class_name = '巨噬细胞'
        self.clusters = []

    def do_cluster(self):
        js_list = []
        for obj in self.obj_list:
            x_avg, y_avg, z_avg = obj.avg_position()
            x, y, z = actual_position(x_avg, y_avg, z_avg)
            js_list.append([x, y, z, obj.score])
        points = np.array([[x, y, z] for x, y, z, _ in js_list])
        dbscan = DBSCAN(eps=2, min_samples=1)
        labels = dbscan.fit_predict(points)
        self.clusters = [None] * (max(labels) + 1)
        for i, label in enumerate(labels):
            if self.clusters[label] is None:
                self.clusters[label] = JS.Cluster()
            self.clusters[label].add_js(js_list[i])

        tmp = []
        for cluster in self.clusters:
            if cluster.js_num > 3:
                tmp.append(cluster)
        self.clusters = tmp

    def __str__(self):
        ret = f"检测到疑似巨噬细胞{len(self.clusters)}簇\n"
        for idx, cluster in enumerate(self.clusters):
            ret += f"第{idx + 1}簇"
            ret += str(cluster)
        return ret


class JC(OCTClass):
    def __init__(self):
        super().__init__()
        self.class_name = '夹层'

    def __str__(self):
        ret = super().__str__()

        return ret


class XS(OCTClass):
    def __init__(self):
        super().__init__()
        self.class_name = '血栓'

    def __str__(self):
        ret = super().__str__()

        return ret


interval = 0.1  # 实际扫描间隔/mm
picture_width = 575  # 画面边长/pixel
actual_width = 10  # 实际长度/mm
picture_s = picture_width ** 2  # 图片画幅面积/pixel
actual_s = actual_width ** 2  # 实际画幅面积/mm^2


def get_size(polygon):
    return (polygon[2][0] - polygon[0][0]) * (polygon[2][1] - polygon[0][1]) * (actual_s / picture_s)


def get_position(polygon):  # 获取obj切片的坐标
    return (polygon[2][0] + polygon[0][0]) / 2, (polygon[2][1] + polygon[0][1]) / 2, polygon[4]


def actual_position(x, y, z):  # 根据像素和扫描间距，计算实际坐标/mm
    return x * (actual_width / picture_width), y * (actual_width / picture_width), z * interval


class OCTObject:
    def __init__(self, label, polygon, score):
        self.label = label
        self.body = [polygon]
        self.score = score
        self.volume = get_size(polygon)
        self.max_square = get_size(polygon)
        self.length = 1 * interval

    def add_slice(self, polygon, score):
        self.body.append(polygon)
        self.volume += get_size(polygon)
        self.max_square = max(self.max_square, get_size(polygon))
        self.score = (self.score * (len(self.body) - 1) + score) / len(self.body)
        self.length += interval

    def avg_position(self):  # 获取obj整体的三维坐标
        ret_x = 0
        ret_y = 0
        ret_z = 0
        for polygon in self.body:
            x, y, z = get_position(polygon)
            ret_x += x
            ret_y += y
            ret_z += z
        ret_x /= len(self.body)
        ret_y /= len(self.body)
        ret_z /= len(self.body)
        return ret_x, ret_y, ret_z


def generate_abstract(result):
    def is_same_obj(last_obj, obj, threshold=25):  # 默认前后两帧的位置差距小于thr像素，则认为是同一个病灶
        if last_obj.label != obj['label']:
            return False
        last_x, last_y, last_z = last_obj.avg_position()
        x, y, z = get_position(obj['poly'])
        if abs(last_x - x) > threshold or abs(last_y - y) > threshold or last_z == z:  # 同一张切片的obj不做融合
            return False
        return True

    def archive_obj(obj):
        if obj.label == 'js':
            js.add_obj(obj)
        elif obj.label == 'jc':
            jc.add_obj(obj)
        else:
            xs.add_obj(obj)

    def try_add_obj(obj):
        for last_obj in obj_list:
            if is_same_obj(last_obj, obj):
                last_obj.add_slice(obj['poly'], obj['score'])
                return
        obj_list.append(OCTObject(obj['label'], obj['poly'], obj['score']))

    def archive_list(cur_z):
        for obj in obj_list:
            if abs(obj.body[-1][4] - cur_z) > 1:  # 清除掉已经不可能连起来的切片，全部当做单独的obj
                archive_obj(obj)
                obj_list.remove(obj)

    js = JS()
    jc = JC()
    xs = XS()
    obj_list = []
    oct_name = result['name']
    oct_result = result['result']

    for slice in oct_result:
        for obj in slice:
            if len(obj_list) == 0:
                obj_list.append(OCTObject(obj['label'], obj['poly'], obj['score']))
                continue

            try_add_obj(obj)
            cur_z = obj['poly'][4]
            archive_list(cur_z)

    for obj in obj_list:
        archive_obj(obj)

    js.do_cluster()
    abstract = f"{js}\n{jc}\n{xs}"

    with open(f"./OCT_Det/result/{oct_name}/{oct_name}.txt", 'w', encoding='utf-8') as f:
        f.write(abstract)
        print(f"save abstract to ./OCT_Det/result/{oct_name}/{oct_name}.txt")

    return abstract


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--result-path', default="./OCT_Det/result/2019_Jul_18_13-29-42/2019_Jul_18_13-29-42.json",
                        required=False, help='Result path')
    args = parser.parse_args()

    with open(args.result_path, 'r') as f:
        result = json.load(f)

    abstract = generate_abstract(result)
