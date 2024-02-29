import json
from argparse import ArgumentParser


class OCTClass:
    def __init__(self):
        self.num = 0
        self.obj_list = []

    def add_obj(self, obj):
        self.obj_list.append(obj)
        self.num += 1


class JS(OCTClass):
    def __init__(self):
        super().__init__()

    def __str__(self):
        result = f"检测到巨噬细胞{self.num}处\n"

        return result


class JC(OCTClass):
    def __init__(self):
        super().__init__()

    def __str__(self):
        result = f"检测到空腔/裂隙{self.num}处\n"
        idx = 1
        for obj in self.obj_list:
            rate = obj.volume / (1000 * 1000) * 100
            result += f"第{idx}处，约占血管截面积{rate}%\n"
            idx += 1

        return result


class XS(OCTClass):
    def __init__(self):
        super().__init__()

    def __str__(self):
        result = f"检测到血栓{self.num}处\n"
        idx = 1
        for obj in self.obj_list:
            rate = obj.volume / (1000 * 1000) * 100
            result += f"第{idx}处，约占血管截面积{rate}%\n"
            idx += 1

        return result

def get_size(polygon):
    return (polygon[2][0] - polygon[0][0]) * (polygon[2][1] - polygon[0][1])


def get_position(polygon):
    return (polygon[2][0] + polygon[0][0]) / 2, (polygon[2][1] + polygon[0][1]) / 2


class OCTObject:

    def __init__(self, label, polygon):
        self.label = label
        self.body = [polygon]
        self.volume = get_size(polygon)
        self.max_square = get_size(polygon)
        self.length = 1

    def add_slice(self, polygon):
        self.body.append(polygon)
        self.volume += get_size(polygon)
        self.max_square = max(self.max_square, get_size(polygon))
        self.length += 1

    def avg_position(self):
        ret_x = 0
        ret_y = 0
        for polygon in self.body:
            x, y = get_position(polygon)
            ret_x += x
            ret_y += y
        ret_x /= len(self.body)
        ret_y /= len(self.body)
        return ret_x, ret_y


def is_same_obj(last_obj, obj):
    if last_obj.label != obj['label']:
        return False
    last_x, last_y = last_obj.avg_position()
    x, y = get_position(obj['poly'])
    if abs(last_x - x) > 50 or abs(last_y - y) > 50:
        return False

    return True


def generate_abstract(oct_result):
    def archive_obj(obj):
        if obj.label == 'js':
            js.add_obj(obj)
        elif obj.label == 'jc':
            jc.add_obj(obj)
        else:
            xs.add_obj(obj)

    js = JS()
    jc = JC()
    xs = XS()
    obj_list = []
    for slice in oct_result:
        for obj in slice:
            if len(obj_list) == 0:
                obj_list.append(OCTObject(obj['label'], obj['poly']))
                continue

            for last_obj in obj_list:
                if is_same_obj(last_obj, obj):
                    last_obj.add_slice(obj['poly'])
                else:
                    archive_obj(last_obj)
                    obj_list.remove(last_obj)
                    obj_list.append(OCTObject(obj['label'], obj['poly']))

    for obj in obj_list:
        archive_obj(obj)

    abstract = f"{js}\n{jc}\n{xs}"

    with open("./OCT_Det/result/abstract.txt", 'w', encoding='utf-8') as f:
        f.write(abstract)
        print("save abstract to ./result/abstract.txt")

    return abstract


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--result-path', default="./OCT_Det/result/result.json", required=False, help='Result path')
    args = parser.parse_args()

    with open(args.result_path, 'r') as f:
        result = json.load(f)

    abstract = generate_abstract(result)

