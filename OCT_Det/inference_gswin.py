import json
import os
import re
from argparse import ArgumentParser
from zipfile import ZipFile

import torch.multiprocessing as mp
from tqdm import tqdm

from OCT_Det.utils import conv2polygon, resize_img, save_gif, NpEncoder
from mmdet.apis import inference_detector, init_detector, show_result_pyplot

from multiprocessing import Process

# 基础路径
raw_path = "./OCT_Det/raw"
result_path = "./OCT_Det/result"


class OCTDetectModel:
    def __init__(self, config, checkpoint, device="cpu", score_thr=0.75):
        self.model = init_detector(config, checkpoint, device=device)
        self.score_thr = score_thr
        self.oct_name = ''
        self.raw_oct = ''
        self.result_oct = ''
        self.result_img_path = ''
        self.img_list = []
        self.result = {}
        self.result_tuple = []
        self.progress_queue = mp.Queue()

    def load_oct(self, file):
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

        # TODO: 在此处嵌入utils编解码模块，file可以是.mp4，pngs.zip，默认zip
        print("Loading file: ", file)
        self.oct_name = os.path.splitext(os.path.basename(file))[0]
        self.raw_oct = os.path.join(raw_path, self.oct_name)
        os.mkdir(self.raw_oct) if not os.path.exists(self.raw_oct) else None
        self.result_oct = os.path.join(result_path, self.oct_name)
        os.mkdir(self.result_oct) if not os.path.exists(self.result_oct) else None
        self.result_img_path = os.path.join(self.result_oct, 'img')
        os.mkdir(self.result_img_path) if not os.path.exists(self.result_img_path) else None
        self.result.update({'name': self.oct_name})
        self.result.update({'result': []})

        if file.endswith('.zip'):
            with ZipFile(file, 'r') as zip_ref:
                zip_ref.extractall(self.raw_oct)

        for f in os.listdir(self.raw_oct):
            if f.endswith('.png'):
                img_path = os.path.join(self.raw_oct, f)
                self.img_list.append(img_path)
                resize_img(img_path, 575)

        self.img_list = sort_by_frame(self.img_list)
        print("loaded_OCT:", self.img_list)

    def reset(self):
        self.img_list = []
        self.result = {}
        self.result_tuple = []

    def inference(self):
        for idx, img in enumerate(tqdm(self.img_list[1:-1], desc='OCT处理进度', unit='img')):
            result_tensor = inference_detector(self.model, img)
            self.result_tuple.append((img, result_tensor))
            conv_result = conv2polygon(self.model, result_tensor, idx, self.score_thr)
            if len(conv_result) > 0:
                self.result['result'].append(conv_result)

    def save_results(self):
        # save json
        with open(os.path.join(self.result_oct, f'{self.oct_name}.json'), 'w') as json_file:
            json_file.write('')
            json.dump(self.result, json_file, indent=4, cls=NpEncoder)
            print(f"save result to: ./{self.result_oct}/{self.oct_name}.json")
        # save png(slow)
        for (img, result) in self.result_tuple:
            show_result_pyplot(self.model, img, result, score_thr=self.score_thr,
                               outfile=os.path.join(self.result_img_path, os.path.basename(img)))
        # save gif
        save_gif(self.result_img_path, os.path.join(self.result_oct, f'{self.oct_name}.gif'))
        print(f"save gif to: ./{self.result_oct}/{self.oct_name}.gif")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--oct', default="./demo/2019_Jul_18_13-29-42.zip", required=False, help='Image zip file')
    parser.add_argument('--config', default="./configs/swin/gswin_oct.py", required=False, help='Config file')
    parser.add_argument('--checkpoint', default="./checkpoints/gswin_transformer.pth", required=False, help='Ckpt')
    parser.add_argument(
        '--device', default='cuda:0', required=False, help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.75, required=False, help='bbox score threshold')
    args = parser.parse_args()

    model = OCTDetectModel(args.config, args.checkpoint, args.device, args.score_thr)
    model.load_oct(args.oct)
    model.inference()
    p = Process(target=model.save_results)
    p.start()
    p.join()
    model.reset()
