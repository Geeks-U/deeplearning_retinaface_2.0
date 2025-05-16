from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
import torch
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, data_dir=Path(r'D:\Data\deeplearning\datasets\widerface\train')):
        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir / 'images'
        self.label_path = self.data_dir / 'label.txt'

        self.samples = self._load_annotations()

    # 返回[relative_path, [[x_lt_bbox, y_lt_bbox, x_rb_bbox, y_rb_bbox, x_ldm, y_ldm, ..., score], ...]]
    def _load_annotations(self):
        samples = []
        with open(self.label_path, 'r') as f:
            lines = f.readlines()

        current_img = None
        current_labels = []

        for line in lines:
            line = line.strip()
            if line.startswith('#'):  # 是图片路径
                if current_img is not None:
                    samples.append((current_img, current_labels))
                current_img = line[2:]  # 去掉 "# "
                current_labels = []
            elif line:  # 是标签行
                parts = list(map(float, line.split()))
                box = parts[:4]
                # 坐标转换为corner形式 x_lt, y_lt, w, h => x_lt, y_lt, x_rb, y_rb
                box = [parts[0], parts[1], parts[0] + parts[2], parts[1] + parts[3]]
                landm = []
                # ldm=-1表示无人脸关键点
                # score=-1表示有人脸但无关键点(ldm置0)
                # score=0表示背景
                # score=1表示有人脸且有关键点
                if parts[4] > 0:
                    for i in range(5):
                        landm += parts[4 + (i * 3):6 + (i * 3)]
                    score = [1]
                else:
                    landm = [0 for i in range(10)]
                    score = [-1]

                label = box + landm + score
                current_labels.append(label)

        # 最后一张图片的内容
        if current_img is not None:
            samples.append((current_img, current_labels))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_relative_path, label = self.samples[idx]
        img_absolute_path = self.image_dir / img_relative_path
        image = Image.open(img_absolute_path)
        label = np.array(label, dtype=np.float32)
        if len(label) == 0:
            return image, np.zeros((0, 15))

        image, label = self._sample_transform(image=image, label=label, input_shape=[320, 320])

        return image, label

    # x_lt, y_lt, x_rb, y_rb
    def _sample_transform(self, image, label, input_shape):
        img_w, img_h = image.size
        input_h, input_w = input_shape
        scale_x = input_w / img_w
        scale_y = input_h / img_h

        image = image.resize((input_w, input_h), Image.BICUBIC)
        new_image = np.array(image, dtype=np.uint8)
        new_image = np.array(new_image, dtype=np.float32) - np.array((123, 117, 104), np.float32)
        new_image = np.transpose(new_image, (2, 0, 1))

        label[:, 0:14:2] *= scale_x / input_w
        label[:, 1:15:2] *= scale_y / input_h

        # 对坐标clip到0~1范围
        label[:, 0:14] = np.clip(label[:, 0:14], 0, 1)

        return new_image, label

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

# 作图工具
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# corner坐标
def draw_image_with_label_px(image, label):
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for item in label:
        # 边界框坐标
        x0, y0 = item[0], item[1]
        x1, y1 = item[2], item[3]
        w, h = x1 - x0, y1 - y0

        # landmarks 坐标
        landmarks = item[4:14]

        # score
        score = item[14]

        # 绘制边界框
        rect = patches.Rectangle((x0, y0), w, h, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        # 绘制 landmarks
        for i in range(0, len(landmarks), 2):
            x_lm, y_lm = landmarks[i], landmarks[i+1]
            ax.plot(x_lm, y_lm, 'bo')

        # 显示 score
        ax.text(x0, y0 - 5, f'{score:.2f}', color='red', fontsize=10)

    plt.axis('off')
    plt.show()

# center坐标
def draw_image_with_label_percent(image, label):
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    img_w, img_h = image.size  # 注意是 (width, height)

    for item in label:
        # 解析中心坐标 + 宽高（百分比 -> 像素）
        cx = item[0] * img_w
        cy = item[1] * img_h
        w = item[2] * img_w
        h = item[3] * img_h

        # 左上角坐标
        x0 = cx - w / 2
        y0 = cy - h / 2

        # landmarks（百分比 -> 像素）
        landmarks = item[4:14]
        for i in range(0, 10, 2):
            x_lm = landmarks[i] * img_w
            y_lm = landmarks[i + 1] * img_h
            ax.plot(x_lm, y_lm, 'bo')  # 蓝色圆点

        # score
        score = item[14]

        # 绘制边界框
        rect = patches.Rectangle((x0, y0), w, h, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        # 绘制 score
        ax.text(x0, y0 - 5, f'{score:.2f}', color='red', fontsize=10)

    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    images_dir = r'D:\Code\Python\deeplearning\data\widerface\train_val\images'
    images_label_path = r'D:\Code\Python\deeplearning\data\widerface\train_val\label.txt'
    img_size = [320, 320]
    data = CustomDataset()
    print(len(data))
    img, label = data[0]
