import os
import time
import argparse
import yaml
import shutil
import cv2
import numpy as np
from object_detection import OBJ_DET_MODELS, draw_bbox
from abc import ABCMeta, abstractmethod


class ClassifierAbs(metaclass=ABCMeta):
    def __init__(self, model_path, input_shape, device):
        self.model_path = model_path
        self.input_shape = input_shape
        self.device = device
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    @staticmethod
    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
        
    def _preprocess(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img = cv2.resize(img, self.input_shape, interpolation=cv2.INTER_LINEAR)
        img = (img/255.0 - self.mean) / self.std
        img = img.transpose([2, 0, 1]).astype(np.float32)
        return img[None]
    
    def _postprocess(self, output):
        cls = np.argmax(output)
        cls_prob = ClassifierAbs.softmax(output)
        return cls, cls_prob
    
    @abstractmethod
    def infer(self, img):
        pass


class ClassiferTorch(ClassifierAbs):
    def __init__(self, model_path, input_shape, device, cfg):
        super().__init__(model_path, input_shape, device)
        import torch
        self.torch = torch
        import torch.backends.cudnn as cudnn
        from lib.models.model import Model
        cudnn.benchmark = cfg["CUDNN"]["BENCHMARK"]
        cudnn.deterministic = cfg["CUDNN"]["DETERMINISTIC"]
        cudnn.enabled = cfg["CUDNN"]["ENABLED"]
        device = 'cuda' if (torch.cuda.is_available() and cfg["GPUS"]) else 'cpu'
        self.device = device
        print("Start infering using device: %s" % device)
        print("Config:", cfg)
        if cfg["MODEL"]["BACKBONE"]["NAME"] == "mobileone":
            cfg["TRAIN"]["PRETRAINED"] = False
            self.model = Model(cfg, training=True)
        else:
            self.model = Model(cfg, training=False)
        weights_path = cfg["TEST"]["WEIGHTS"]
        if not weights_path:
            raise Exception("Please specify path to model weights in config file!")
        weights = torch.load(weights_path, map_location=device)
        self.model.load_state_dict(weights['state_dict'] if 'state_dict' in weights else weights)
        if cfg["MODEL"]["BACKBONE"]["NAME"] == "mobileone":
            self.model.backbone = self.model.backbone.reparameterize_model()
        self.model.to(device)
        self.model.eval()

    def infer(self, img):
        np_input = self._preprocess(img)
        inp = self.torch.Tensor(np_input).float().to(self.device)
        speed = list()
        for _ in range(10):
            begin = time.time()
            output = self.model(inp)[0]
            speed.append(time.time()-begin)
        np_output = output.cpu().detach().numpy() if output.requires_grad else output.cpu().numpy()
        cls, cls_prob = self._postprocess(np_output)
        return cls, cls_prob, np.mean(speed)

    def infer_batch(self, imgs):
        for i in range(len(imgs)):
            imgs[i] = self._preprocess(imgs[i])
        inp = np.concatenate(imgs, axis=0)
        inp = self.torch.Tensor(inp).float().to(self.device)
        speed = list()
        for _ in range(10):
            begin = time.time()
            outputs = self.model(inp)
            speed.append(time.time()-begin)
        clss, cls_probs = list(), list()
        for output in outputs:
            np_output = output.cpu().detach().numpy() if output.requires_grad else output.cpu().numpy()
            cls, cls_prob = self._postprocess(np_output)
            clss.append(cls)
            cls_probs.append(cls_prob)
        return clss, cls_probs, np.mean(speed)/len(imgs)


class ClassiferOnnx(ClassifierAbs):
    def __init__(self, model_path, input_shape, device):
        super().__init__(model_path, input_shape, device)
        import onnxruntime
        print("Start infering using device: %s" % device)
        self.ort_session = onnxruntime.InferenceSession(model_path)
        self.input_name = self.ort_session.get_inputs()[0].name

    def infer(self, img):
        inp = self._preprocess(img)
        speed = list()
        for _ in range(10):
            begin = time.time()
            output = self.ort_session.run(None, {self.input_name: inp})[0][0]
            speed.append(time.time()-begin)
        cls, cls_prob = self._postprocess(output)
        return cls, cls_prob, np.mean(speed)

    def infer_batch(self, imgs):
        for i in range(len(imgs)):
            imgs[i] = self._preprocess(imgs[i])
        inp = np.concatenate(imgs, axis=0)
        speed = list()
        for _ in range(10):
            begin = time.time()
            np_outputs = self.ort_session.run(None, {self.input_name: inp})[0]
            speed.append(time.time()-begin)
        clss, cls_probs = list(), list()
        for np_output in np_outputs:
            cls, cls_prob = self._postprocess(np_output)
            clss.append(cls)
            cls_probs.append(cls_prob)
        return clss, cls_probs, np.mean(speed)/len(imgs)


def main(opt):
    if opt.engine == "torch":
        with open(opt.config, "r") as stream:
            try:
                cfg = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
                quit()

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
        if cfg["GPUS"]:
            os.environ["CUDA_VISIBLE_DEVICES"] = cfg["GPUS"]

        engine = ClassiferTorch("doesnt_matter",
                                cfg["MODEL"]["INPUT_SHAPE"],
                                "doesnt_matter",
                                cfg)
    elif opt.engine == "onnx":
        engine = ClassiferOnnx(opt.model, opt.input_shape, opt.device)
    else:
        raise NotImplementedError("Engine %s is not supported!" % opt.engine)
    det_engine = None
    if opt.det == "yolov5":
        det_engine = OBJ_DET_MODELS["yolov5"](opt.det_model, opt.det_cls,
            opt.det_num_cls, opt.det_engine, opt.det_input, opt.device, 
            opt.det_conf, opt.det_iou)
    else:
        print("Do not use object detection")

    if opt.src == "image":
        for img in opt.src_path:
            print("Image:", img)
            img = cv2.imread(img)
            cls, cls_prob, latency = engine.infer(img)

            print("Result:")
            cls_name = opt.cls[cls] if opt.cls else cls
            print("Class: %i (%s), score: %.4f" % (cls, cls_name, cls_prob[cls]))
            print("Classes probability:", cls_prob)
            print("Latency: %.4f, FPS: %.2f" % (latency, 1/latency))
    elif opt.src == "folder":
        for fd in opt.src_path:
            print("Folder:", fd)
            for fn in os.listdir(fd):
                print("File:", fn)
                fp = os.path.join(fd, fn)
                img = cv2.imread(fp)
                try:
                    cls, cls_prob, latency = engine.infer(img)
                except Exception as e:
                    print(e)
                    print("Ignoring %s..." % fn)
                    continue
                # print("Result:")
                cls_name = opt.cls[cls] if opt.cls else cls
                print("Class: %i (%s), score: %.4f" % (cls, cls_name, cls_prob[cls]))
                # print("Classes probability:", cls_prob)
                print("Latency: %.4f, FPS: %.2f" % (latency, 1/latency))
                if opt.dst_path:
                    lbl_dir = os.path.join(opt.dst_path, cls_name)
                    os.makedirs(lbl_dir, exist_ok=True)
                    print("Copying %s to %s..." % (fn, lbl_dir))
                    shutil.copy(fp, lbl_dir)
    elif opt.src == "video":
        if det_engine is None:
            print("Please specify object detector for video source!")
            quit()
        for vid in opt.src_path:
            print("Video:", vid)
            cap = cv2.VideoCapture(vid)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            ext = "." + vid.split(".")[-1]
            dst_path = vid.replace(ext, "_result" + ext)
            writer = cv2.VideoWriter(dst_path, cv2.VideoWriter_fourcc(*'mp4v'), 
                10, (width, height))
            if (cap.isOpened()== False):
                print("Error opening video stream or file")
            count = 0
            while cap.isOpened():
                count += 1
                print("Processing frame %i/%i" % (count, length))
                ret, frame = cap.read()
                if not ret: break
                boxes = det_engine.infer(frame.copy())
                if opt.batch:
                    imgs = list()
                    for box in boxes:
                        x1, y1, x2, y2 = box[:4]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        obj = frame[y1:y2, x1:x2]
                        imgs.append(obj)
                    clss, cls_probs, latency = engine.infer_batch(imgs)
                    print("Latency: %.4f" % latency)
                    for i in range(len(clss)):
                        x1, y1, x2, y2 = boxes[i][:4]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        cls, cls_prob = clss[i], cls_probs[i]
                        cls_name = opt.cls[cls] if opt.cls else cls
                        cls_str = "%s %.2f" % (cls_name, cls_prob[cls])
                        frame = draw_bbox(frame, cls_str, (x1, y1), (x2, y2))
                else:
                    for box in boxes:
                        x1, y1, x2, y2 = box[:4]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        obj = frame[y1:y2, x1:x2]
                        cls, cls_prob, latency = engine.infer(obj)
                        print("Latency: %.4f" % latency)
                        cls_name = opt.cls[cls] if opt.cls else cls
                        cls_str = "%s %.2f" % (cls_name, cls_prob[cls])
                        frame = draw_bbox(frame, cls_str, (x1, y1), (x2, y2))
                cv2.imshow("Test", frame)
                writer.write(frame)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            cap.release()
            writer.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src',
                        type=str,
                        default='image',
                        help='source type (image, folder, video)')
    parser.add_argument('--src-path',
                        type=str,
                        nargs="+",
                        default='examples/images/cat.jpeg',
                        help='path to source')
    parser.add_argument('--dst-path',
                        type=str,
                        default=None,
                        help='path to save labels (folder src type only)')
    parser.add_argument('--engine',
                        type=str,
                        default='onnx',
                        help='engine type (onnx, mnn, torch)')
    parser.add_argument('--model',
                        type=str,
                        default='weights/dogsvscats_shufflenetv2_none_linearcls_10eps.onnx',
                        help='path to model weights')
    parser.add_argument('--input-shape', 
                        nargs='+',
                        type=int, 
                        default=(224, 224), 
                        help='input shape for classification model')
    parser.add_argument('--cls',
                        type=str,
                        nargs="+",
                        help='class names for classification')
    parser.add_argument('--batch', action='store_true', 
                        help='use batch for classification')
    parser.add_argument('--config',
                        type=str,
                        default='configs/customds/dogsvscats_shufflenetv2_none_linearcls_10eps.yaml',
                        help='path to config file (only for torch engine)')
    parser.add_argument('--det',
                        type=str,
                        default=None,
                        help='object detection model, leave it blank to ignore detection')
    parser.add_argument('--det-engine',
                        type=str,
                        default='onnx',
                        help='engine type for object detection (onnx, mnn, torch)')
    parser.add_argument('--det-model',
                        type=str,
                        default='weights/yolov5.onnx',
                        help='path to object detection model weights')
    parser.add_argument('--det-cls',
                        type=int,
                        nargs="+",
                        help='classes for object detection')
    parser.add_argument('--det-num-cls',
                        type=int,
                        help='numnber of classes for object detection model')
    parser.add_argument('--det-input',
                        type=int,
                        nargs="+",
                        default=[640, 640],
                        help='object detection input shape')
    parser.add_argument('--det-conf',
                        type=float,
                        default=0.25,
                        help='object detection conf threshold')
    parser.add_argument('--det-iou',
                        type=float,
                        default=0.55,
                        help='object detection iou threshold')
    parser.add_argument('--device',
                        type=str,
                        default='cpu',
                        help='device to run infer on')
    
    opt = parser.parse_args()
    main(opt)
