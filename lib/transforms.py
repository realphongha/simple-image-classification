import random
import math
import numpy as np


def random_erasing(img, p, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0):
    if random.random() > p:
        return
    try:
        h, w = img.shape[:2]
        for _ in range(10):
            s = random.uniform(*scale)
            r = np.exp(random.uniform(*ratio))
            e_size = h*w*s
            e_w = math.sqrt(e_size/r)
            e_h = e_w * r
            e_w, e_h = round(e_w), round(e_h)
            if e_w >= w or e_h >= h:
                continue
            e_x0 = random.randint(0, w-e_w-1)
            e_y0 = random.randint(0, h-e_h-1)
            if type(value) == int:
                value = [value] * 3
            img[e_y0:e_y0+e_h, e_x0:e_x0+e_w] = np.array(value)
            return
    except Exception as e:
        print(e)
        print(s, r, e_size, e_w, e_h)
        quit()
