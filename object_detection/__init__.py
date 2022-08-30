import cv2
from .yolov5 import get_yolov5

OBJ_DET_MODELS = {
    "yolov5": get_yolov5
}


def draw_bbox(image, text, start, end):
    image = cv2.rectangle(image, start, end, (0, 0, 255), 2)
    start = (start[0], start[1]-5)
    image = cv2.putText(image, text, start, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
        (0, 0, 255), 1, cv2.LINE_AA)
    return image
