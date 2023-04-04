import time
import numpy as np
import cv2
from abc import ABCMeta, abstractmethod


class ClassifierAbs(metaclass=ABCMeta):
    def __init__(self, model_path, input_shape, device, gray=False):
        self.model_path = model_path
        self.input_shape = input_shape
        self.device = device
        self.gray = gray
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    @staticmethod
    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def _preprocess(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.input_shape, interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32)
        if self.gray:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = np.expand_dims(img, axis=2)
            img = (img/255.0 - 0.5) * 2
            print(img.shape)
        else:
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
    def __init__(self, model_path, input_shape, device, gray, cfg, compile):
        super().__init__(model_path, input_shape, device, gray)
        import torch
        self.torch = torch
        import torch.backends.cudnn as cudnn
        from lib.models.model import Model
        cudnn.benchmark = cfg["CUDNN"]["BENCHMARK"]
        cudnn.deterministic = cfg["CUDNN"]["DETERMINISTIC"]
        cudnn.enabled = cfg["CUDNN"]["ENABLED"]
        self.compile = compile
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

        # compiles model
        if self.compile == "default":
            print(f"Compiling model in {self.compile} mode...")
            self.model = torch.compile(self.model)
        elif self.compile in ("reduce-overhead", "max-autotune"):
            print(f"Compiling model in {self.compile} mode...")
            self.model = torch.compile(self.model, mode=self.compile)


    def infer(self, img, test_time=50):
        np_input = self._preprocess(img)
        inp = self.torch.Tensor(np_input).float().to(self.device)
        speed = list()
        for _ in range(test_time):
            begin = time.time()
            output = self.model(inp)[0]
            speed.append(time.time()-begin)
        np_output = output.cpu().detach().numpy() if output.requires_grad else output.cpu().numpy()
        cls, cls_prob = self._postprocess(np_output)
        return cls, cls_prob, np.mean(speed)

    def infer_batch(self, imgs, test_time=50):
        for i in range(len(imgs)):
            imgs[i] = self._preprocess(imgs[i])
        inp = np.concatenate(imgs, axis=0)
        inp = self.torch.Tensor(inp).float().to(self.device)
        speed = list()
        for _ in range(test_time):
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
    def __init__(self, model_path, input_shape, device, gray):
        super().__init__(model_path, input_shape, device, gray)
        import onnxruntime
        print("Start infering using device: %s" % device)
        if device == "cuda":
            providers = ["CUDAExecutionProvider"]
        elif device == "cpu":
            providers = ["CPUExecutionProvider"]
        else:
            raise NotImplementedError(f"Device {device} is not implemented!")
        self.ort_session = onnxruntime.InferenceSession(model_path, providers=providers)
        self.input_name = self.ort_session.get_inputs()[0].name

    def infer(self, img, test_time=50):
        inp = self._preprocess(img)
        speed = list()
        for _ in range(test_time):
            begin = time.time()
            output = self.ort_session.run(None, {self.input_name: inp})[0][0]
            speed.append(time.time()-begin)
        cls, cls_prob = self._postprocess(output)
        return cls, cls_prob, np.mean(speed)

    def infer_batch(self, imgs, test_time=50):
        for i in range(len(imgs)):
            imgs[i] = self._preprocess(imgs[i])
        inp = np.concatenate(imgs, axis=0)
        speed = list()
        for _ in range(test_time):
            begin = time.time()
            np_outputs = self.ort_session.run(None, {self.input_name: inp})[0]
            speed.append(time.time()-begin)
        clss, cls_probs = list(), list()
        for np_output in np_outputs:
            cls, cls_prob = self._postprocess(np_output)
            clss.append(cls)
            cls_probs.append(cls_prob)
        return clss, cls_probs, np.mean(speed)/len(imgs)

