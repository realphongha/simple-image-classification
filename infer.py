import os
import time
import argparse
import yaml
import cv2
import numpy as np
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
        self.model = Model(cfg, training=False)
        weights_path = cfg["TEST"]["WEIGHTS"]
        if not weights_path:
            raise Exception("Please specify path to model weights in config file!")
        weights = torch.load(weights_path, map_location=device)
        self.model.load_state_dict(weights['state_dict'] if 'state_dict' in weights else weights)
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
    
    img = cv2.imread(opt.image)
    cls, cls_prob, latency = engine.infer(img)

    print("Result:")
    print("Class: %i, score: %.4f" % (cls, cls_prob[cls]))
    print("Classes probability:", cls_prob)
    print("Latency: %.2f, FPS: %.2f" % (latency, 1/latency))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image',
                        type=str,
                        default='examples/images/cat.jpeg',
                        help='path to image')
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
    parser.add_argument('--device',
                        type=str,
                        default='cpu',
                        help='device to run infer on')
    parser.add_argument('--config',
                        type=str,
                        default='configs/customds/dogsvscats_shufflenetv2_none_linearcls_10eps.yaml',
                        help='path to config file (only for torch engine)')
    opt = parser.parse_args()
    main(opt)
