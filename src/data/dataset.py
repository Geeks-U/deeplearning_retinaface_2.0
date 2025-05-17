from pathlib import Path
import copy

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import cv2

cfg_dataset_default = {
    'image_dir': r'D:\Data\deeplearning\datasets\widerface\train\images',
    'label_path': r'D:\Data\deeplearning\datasets\widerface\train\label.txt',
    'image_input_size': [320, 320]
}

class CustomDataset(Dataset):
    def __init__(self, cfg_dataset=None):
        self.cfg = copy.deepcopy(cfg_dataset_default)
        if cfg_dataset is not None:
            self.cfg.update(cfg_dataset)
        self.image_dir = Path(self.cfg['image_dir'])
        self.label_path = Path(self.cfg['label_path'])
        self.image_input_size = self.cfg['image_input_size']

        self.samples = self._load_annotations()

    # 输入: relative_path, [[x_lt_bbox, y_lt_bbox, w_bbox, h_bbox, x0_ldm, y0_ldm, score0_ldm, ..., score], ...]
    # 输出: relative_path, [[x_lt_bbox, y_lt_bbox, x_rb_bbox, y_rb_bbox, x_ldm, y_ldm, ..., score], ...]
    # 有人脸有关键点 score=1
    # 有人脸无关键点 score=-1，ldm值为-1表示无人脸关键点，则将对应坐标置0
    # 无人脸 score=0
    def _load_annotations(self):
        samples = []
        with open(self.label_path, 'r') as f:
            lines = f.readlines()

        current_img = None
        current_labels = []

        for line in lines:
            line = line.strip()
            if line.startswith('#'):
                if current_img is not None:
                    samples.append((current_img, current_labels))
                current_img = line[2:]
                current_labels = []
            elif line:
                parts = list(map(float, line.split()))
                box = parts[:4]
                box = [parts[0], parts[1], parts[0] + parts[2], parts[1] + parts[3]]
                landm = []
                if parts[4] > 0:
                    for i in range(5):
                        landm += parts[4 + (i * 3):6 + (i * 3)]
                    score = [1]
                else:
                    landm = [0 for i in range(10)]
                    score = [-1]

                label = box + landm + score
                current_labels.append(label)

        if current_img is not None:
            samples.append((current_img, current_labels))

        return samples

    def __len__(self):
        return len(self.samples)

    # 输入: 索引
    # 输出: np.array类型的image, label
    def __getitem__(self, idx):
        img_relative_path, label = self.samples[idx]
        img_absolute_path = self.image_dir / img_relative_path
        image = Image.open(img_absolute_path)
        label = np.array(label, dtype=np.float32)
        if len(label) == 0:
            return image, np.zeros((0, 15))

        # image, label = self._sample_transform(image=image, label=label, input_shape=self.image_input_size)
        image, label = self.get_random_data(image, label, self.image_input_size)
        image = np.transpose(self.preprocess_input(np.array(image, np.float32)), (2, 0, 1))

        return image, label

    # 输入: image(PIL打开), label(np.array类型)(corner_px)
    # 输出: np.array类型的image, label(corner_percent)
    def _sample_transform(self, image, label, input_shape):
        img_w, img_h = image.size
        input_h, input_w = input_shape

        # 图像调整
        new_image = image.resize((input_w, input_h), Image.BICUBIC)
        new_image = np.array(new_image, dtype=np.uint8)
        new_image = np.array(new_image, dtype=np.float32) - np.array((123, 117, 104), np.float32)
        new_image = np.transpose(new_image, (2, 0, 1))

        # 标签同步调整
        label[:, 0:14:2] /= img_w
        label[:, 1:15:2] /= img_h
        # label坐标clip到0~1范围
        label[:, 0:14] = np.clip(label[:, 0:14], 0, 1)

        return new_image, label

    def get_random_data(self, image, targets, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.4):
        iw, ih  = image.size
        h, w    = input_shape
        box     = targets

        #------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        #------------------------------------------#
        new_ar = w/h * self.rand(1-jitter,1+jitter)/self.rand(1-jitter,1+jitter)
        scale = self.rand(0.25, 3.25)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)

        #------------------------------------------#
        #   将图像多余的部分加上灰条
        #------------------------------------------#
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image = new_image

        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
        flip = self.rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        image_data      = np.array(image, np.uint8)
        #---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        #---------------------------------#
        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        #---------------------------------#
        #   将图像转到HSV上
        #---------------------------------#
        hue, sat, val   = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype           = image_data.dtype
        #---------------------------------#
        #   应用变换
        #---------------------------------#
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        #---------------------------------#
        #   对真实框进行调整
        #---------------------------------#
        if len(box)>0:
            np.random.shuffle(box)
            box[:, [0,2,4,6,8,10,12]] = box[:, [0,2,4,6,8,10,12]]*nw/iw + dx
            box[:, [1,3,5,7,9,11,13]] = box[:, [1,3,5,7,9,11,13]]*nh/ih + dy
            if flip:
                box[:, [0,2,4,6,8,10,12]] = w - box[:, [2,0,6,4,8,12,10]]
                box[:, [5,7,9,11,13]]     = box[:, [7,5,9,13,11]]

            center_x = (box[:, 0] + box[:, 2])/2
            center_y = (box[:, 1] + box[:, 3])/2

            box = box[np.logical_and(np.logical_and(center_x>0, center_y>0), np.logical_and(center_x<w, center_y<h))]

            box[:, 0:14][box[:, 0:14]<0] = 0
            box[:, [0,2,4,6,8,10,12]][box[:, [0,2,4,6,8,10,12]]>w] = w
            box[:, [1,3,5,7,9,11,13]][box[:, [1,3,5,7,9,11,13]]>h] = h

            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box

        # 将不含有人脸关键点的box的landm置为0
        box[:,4:-1][box[:,-1]==-1]=0
        # 归一化
        box[:, [0,2,4,6,8,10,12]] /= w
        box[:, [1,3,5,7,9,11,13]] /= h
        box_data = box
        return image_data, box_data

    def preprocess_input(self, image):
        image -= np.array((104, 117, 123),np.float32)
        return image
    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a

# 自定义批次处理函数
def detection_collate(batch):
    images  = []
    targets = []
    for img, box in batch:
        if len(box)==0:
            continue
        images.append(img)
        targets.append(box)
    images = np.array(images)
    return images, targets
