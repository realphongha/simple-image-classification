import os
import argparse
import yaml
import shutil
import cv2
from object_detection import OBJ_DET_MODELS, draw_bbox
from lib.standalone_engine import *


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
                                "doesnt_matter", opt.gray,
                                cfg, opt.compile)
    elif opt.engine == "onnx":
        engine = ClassiferOnnx(opt.model, opt.input_shape, opt.device,
                               opt.gray)
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
            cls_name = opt.cls[cls] if opt.cls else str(cls)
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
                cls_name = opt.cls[cls] if opt.cls else str(cls)
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
                    clss, cls_probs, latency = engine.infer_batch(imgs, 1)
                    print("Latency: %.4f" % latency)
                    for i in range(len(clss)):
                        x1, y1, x2, y2 = boxes[i][:4]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        cls, cls_prob = clss[i], cls_probs[i]
                        cls_name = opt.cls[cls] if opt.cls else str(cls)
                        cls_str = "%s %.2f" % (cls_name, cls_prob[cls])
                        frame = draw_bbox(frame, cls_str, (x1, y1), (x2, y2))
                else:
                    for box in boxes:
                        x1, y1, x2, y2 = box[:4]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        obj = frame[y1:y2, x1:x2]
                        cls, cls_prob, latency = engine.infer(obj, 1)
                        print("Latency: %.4f" % latency)
                        cls_name = opt.cls[cls] if opt.cls else str(cls)
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
    parser.add_argument('--gray',
                        action="store_true",
                        default=False,
                        help='convert image to grayscale for processing or not?')
    parser.add_argument('--compile',
                        type=str,
                        default='no',
                        help='Pytorch 2.0 compile, options: default, reduce-overhead, max-autotune, no')
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
