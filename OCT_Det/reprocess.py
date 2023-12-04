

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
        return f"检测到巨噬细胞{self.num}处"


class JC(OCTClass):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return f"检测到空腔/裂隙{self.num}处"


class XS(OCTClass):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return f"检测到血栓{self.num}处"


def get_size(polygon):
    return (polygon[2][0] - polygon[0][0]) * (polygon[2][1] - polygon[0][1])


def get_position(polygon):
    return (polygon[2][0] + polygon[0][0]) / 2, (polygon[2][1] + polygon[0][1]) / 2


class OCTObject:

    def __init__(self, label, polygon):
        self.label = label
        self.body = [polygon]
        self.volume = get_size(polygon)

    def add_slice(self, polygon):
        self.body.append(polygon)
        self.volume += get_size(polygon)

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

    abstract = f"{js}; {jc}; {xs}。"
    return abstract
