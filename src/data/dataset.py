from pathlib import Path
import copy

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import cv2

cfg_dataset_default = {
    'image_dir': r'D:\Data\deeplearning\datasets\widerface\train\images',
    'label_path': r'D:\Data\deeplearning\datasets\widerface\train\label.txt',
    'input_image_size': [320, 320],
    'augment': False
}

class CustomDataset(Dataset):
    def __init__(self, cfg_dataset=None):
        self.cfg = copy.deepcopy(cfg_dataset_default)
        if cfg_dataset is not None:
            self.cfg.update(cfg_dataset)
        self.image_dir = Path(self.cfg['image_dir'])
        self.label_path = Path(self.cfg['label_path'])
        self.input_image_size = self.cfg['input_image_size']

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

        image, label = self._sample_transform(image=image, label=label, input_shape=self.input_image_size)

        return image, label

    # 输入: image(PIL打开), label(np.array类型)(corner_px)
    # 输出: np.array类型的image, label(corner_percent)
    def _sample_transform(self, image, label, input_shape):
        img_w, img_h = image.size
        input_h, input_w = input_shape

        # 是否进行数据增强
        if self.cfg['augment']:
            pass

        # 图像缩放 + 减去均值 + 转置维度
        new_image = np.array(image.resize((input_w, input_h), Image.BICUBIC), dtype=np.float32)
        mean = np.array([123.0, 117.0, 104.0], dtype=np.float32)
        new_image -= mean
        new_image = new_image.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)

        # 数据清洗
        np.random.shuffle(label)

        # 步骤1：筛除非法中心点
        center = (label[:, 0:2] + label[:, 2:4]) * 0.5
        inside_mask = (
            (center[:, 0] > 0) & (center[:, 0] < img_w) &
            (center[:, 1] > 0) & (center[:, 1] < img_h)
        )
        label = label[inside_mask]

        # 步骤2：筛除非法宽高（在中心合法的前提下）
        size = label[:, 2:4] - label[:, 0:2]
        valid_size_mask = (size[:, 0] > 1) & (size[:, 1] > 1)
        label = label[valid_size_mask]

        # 坐标归一化并裁剪
        coords = label[:, 0:14]
        coords[:, 0:14:2] /= img_w
        coords[:, 1:14:2] /= img_h
        # 原地操作
        np.clip(coords, 0, 1, out=coords)

        return new_image, label

# 自定义批次处理函数(默认会将批次label对齐, 该项目每个图象label数量可能不同导致对不齐)
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
