import os
from zipfile import ZipFile

from OCT_Det.utils import conv2polygon, resize_img
from mmdet.apis import inference_detector, init_detector

raw_path = "/raw"


class OCTDetectModel:
    def __init__(self, config, checkpoint, device="cpu", score_thr=0.3):
        self.model = init_detector(config, checkpoint, device=device)
        self.score_thr = score_thr
        self.img_list = []
        self.result = []

    def load_oct(self, file):
        # TODO: 在此处嵌入utils编解码模块，file可以是.mp4，pngs.zip，先按照zip
        print("Loading file: ", file)
        if file.endswith('.zip'):
            with ZipFile(file, 'r') as zip_ref:
                zip_ref.extractall(raw_path)
        self.img_list = sorted([os.path.join(raw_path, f) for f in os.listdir(raw_path) if f.endswith('.png')])
        resize_img(self.img_list)
        print("loaded_OCT:", self.img_list)

    def reset(self):
        self.img_list = []
        self.result = []

    def inference(self):
        for img in self.img_list[1:-1]:
            result_tensor = inference_detector(self.model, img)
            self.result.append(conv2polygon(self.model, result_tensor, self.score_thr))
