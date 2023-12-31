import os
from zipfile import ZipFile
from argparse import ArgumentParser

from OCT_Det.utils import conv2polygon, resize_img, save_gif
from mmdet.apis import inference_detector, init_detector, show_result_pyplot

raw_path = "./raw"
result_path = "./result"
gif_path = "./result/result.gif"


class OCTDetectModel:
    def __init__(self, config, checkpoint, device="cpu", score_thr=0.3):
        self.model = init_detector(config, checkpoint, device=device)
        self.score_thr = score_thr
        self.img_list = []
        self.result = []
        self.result_tuple = []

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
        self.result_tuple = []

    def inference(self):
        for img in self.img_list[1:-1]:
            result_tensor = inference_detector(self.model, img)
            self.result_tuple.append((img, result_tensor))
            self.result.append(conv2polygon(self.model, result_tensor, self.score_thr))

    async def save_results(self):
        for (img, result) in self.result_tuple:
            show_result_pyplot(self.model, img, result, score_thr=self.score_thr,
                               outfile=os.path.join(result_path, img.split("raw\\")[-1]))
        save_gif(result_path, gif_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--oct', default="../demo/demo.zip", required=False, help='Image zip file')
    parser.add_argument('--config', default="../configs/swin/gswin_oct.py", required=False, help='Config file')
    parser.add_argument('--checkpoint', default="../checkpoints/gswin_transformer.pth", required=False, help='Ckpt')
    parser.add_argument(
        '--device', default='cuda:0', required=False, help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, required=False, help='bbox score threshold')
    args = parser.parse_args()

    model = OCTDetectModel(args.config, args.checkpoint, args.device, args.score_thr)
    model.load_oct(args.oct)
    model.inference()
    model.save_results()
    model.reset()
