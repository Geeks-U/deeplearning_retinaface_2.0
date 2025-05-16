import os
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch.utils.data as data


class CustomWiderFaceDataset(data.Dataset):
    def __init__(self,
                 images_dir=Path(r'D:\Data\deeplearning\datasets\widerface\train') / 'images',
                 images_label_path=Path(r'D:\Data\deeplearning\datasets\widerface\train') / 'label.txt',
                 input_image_size=[320, 320],
                 transform=None):
        self.images_dir = images_dir
        self.images_label_path = images_label_path
        self.input_image_size = input_image_size
        self.transform = transform
        self.validate_custom_widerface_params()
        self.images_path, self.labels = self._load_data()

    # 参数验证
    def validate_custom_widerface_params(self):
        # 验证 images_dir 是否是有效的目录
        if not os.path.isdir(self.images_dir):
            raise ValueError(f"Invalid directory: {self.images_dir}")

        # 验证 images_label_path 是否是有效的文件路径
        if not os.path.isfile(self.images_label_path):
            raise ValueError(f"Invalid file path: {self.images_label_path}")

        # 检查输入图像尺寸是否为包含两个正整数的列表
        if (
            not isinstance(self.input_image_size, list) or
            len(self.input_image_size) != 2 or
            not all(isinstance(dim, int) and dim > 0 for dim in self.input_image_size)
        ):
            raise ValueError(f"Invalid input_image_size: {self.input_image_size}. It must be a list of two positive integers.")

        print("CustomWiderFaceDataset configuration is valid.")

    # 数据加载
    def _load_data(self):
        """
        从指定路径加载图像文件和对应的标签数据。

        该函数读取包含图像路径和标签信息的文件，并根据文件内容生成图像路径和标签列表。标签包括图像中目标的边界框（bounding box）、五个关键点（landmarks）和目标的置信度分数。每个标签由20个浮动值组成，其中包括边界框的四个坐标、每个关键点的三个坐标值，以及最终的置信度分数。

        文件格式如下：
        - 每个图像的标签以`#`开头的行开始，接下来的是该图像对应的目标标签。
        - 标签行由20个浮动值组成，格式如下：
            - 第一个四个值表示边界框的坐标（xmin, ymin, width, height）。
            - 接下来的15个值表示五个关键点，每个关键点由x、y坐标和一个置信度分数组成。
            - 最后一个值表示目标的置信度分数。

        过程：
        1. 读取文件并逐行解析。
        2. 遇到`#`符号时，记录图像路径并开始处理新的图像标签。
        3. 每行非`#`的标签数据被分解成浮动值，并转换为边界框坐标、关键点坐标及分数。
        4. 所有的图像路径和标签最终被整理成列表返回。

        返回:
            - `images_path`: 包含所有图像完整路径的列表。
            - `labels`: 对应图像的标签列表，每个标签包括目标的边界框(左上角(x, y), 右下角(x, y))、关键点和置信度分数。
        """
        images_path = []
        labels = []

        with open(self.images_label_path, 'r') as f:
            lines = f.readlines()

        lines_iter = iter(lines)
        current_labels = []

        for line in lines_iter:
            line = line.strip()

            if not line:
                continue

            if line.startswith('#'):
                # 新图像开始：如果有旧图像的标签，先保存
                if current_labels:
                    labels.append(current_labels)
                    current_labels = []

                relative_path = line[1:].strip()
                full_path = os.path.join(self.images_dir, relative_path)
                images_path.append(full_path)

            else:
                parts = list(map(float, line.split()))
                if len(parts) != 20:
                    raise ValueError(f"Expected 20 float values for label, got {len(parts)}: {line}")

                box = parts[:4]
                # 坐标转换为corner形式
                box = [parts[0], parts[1], parts[0] + parts[2], parts[1] + parts[3]]
                landm = []
                for i in range(5):
                    landm += parts[4 + (i * 3):6 + (i * 3)]
                score = [parts[19]]
                label = box + landm + score
                current_labels.append(label)

        # 文件结尾，别忘了最后一组标签
        if current_labels:
            labels.append(current_labels)

        return images_path, labels

    def __len__(self):
        return len(self.images_path)

    # 数据获取
    def __getitem__(self, index):
        image = Image.open(self.images_path[index])
        label = np.array(self.labels[index], dtype=np.float32)
        if len(label) == 0:
            return image, np.zeros((0, 15))
        # label处理 没有人脸关键点的数据标记为-1，否则为1
        for l in label:
            if l[4] < 0:
                l[-1] = -1
            else:
                l[-1] = 1
        # image, label = self.get_random_data(image, label, self.input_image_size)
        image, label = self.get_random_data_demo(image, label, self.input_image_size)
        image = np.transpose(self.preprocess_input(np.array(image, np.float32)), (2, 0, 1))
        return image, label
    # 图片增强 并同步标签
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

    def get_random_data_demo(self, image, targets, input_shape):
        iw, ih = image.size
        h, w = input_shape
        box = targets

        # 直接等比例缩放到目标大小
        image = image.resize((w, h), Image.BICUBIC)
        image_data = np.array(image, np.uint8)

        # 真实框也按缩放比例调整
        if len(box) > 0:
            box[:, [0,2,4,6,8,10,12]] = box[:, [0,2,4,6,8,10,12]] * w / iw
            box[:, [1,3,5,7,9,11,13]] = box[:, [1,3,5,7,9,11,13]] * h / ih

            # 将不含有人脸关键点的 box 的 landm 置为 0
            box[:, 4:-1][box[:, -1] == -1] = 0

            # 归一化
            box[:, [0,2,4,6,8,10,12]] /= w
            box[:, [1,3,5,7,9,11,13]] /= h

        box_data = box
        return image_data, box_data


    # label数据格式：第1组坐标为图片左上角，第2组坐标为图片右下角，第3-7组为landm，最后一个数据是score
    def _co_transform(self, img, label, img_size, flip_prob=0.7, crop_prob=0.5, brightness_factor=0.2, contrast_factor=0.2):
        original_w, original_h = img.size
        target_w, target_h = img_size

        # 0. Resize 图像
        img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
        scale_x = target_w / original_w
        scale_y = target_h / original_h

        # 缩放 x 和 y 坐标
        label[:, [0, 2, 4, 6, 8, 10, 12, 14]] *= scale_x  # 所有 x 坐标
        label[:, [1, 3, 5, 7, 9, 11, 13]] *= scale_y      # 所有 y 坐标

        # 1. 随机水平翻转
        if random.random() < flip_prob:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            for obj in label:
                # 翻转 bbox
                x1 = target_w - obj[2]
                x2 = target_w - obj[0]
                obj[0], obj[2] = x1, x2

                # 翻转 landmarks 的 x 坐标
                for i in range(4, 14, 2):
                    obj[i] = target_w - obj[i]

        # 2. 对无效的 landmark 清零（如果 score == -1）
        label[label[:, -1] == -1, 4:14] = 0

        # 3. 归一化
        label[:, [0, 2, 4, 6, 8, 10, 12, 14]] /= target_w  # x 归一化
        label[:, [1, 3, 5, 7, 9, 11, 13]] /= target_h      # y 归一化

        return img, label
    def preprocess_input(self, image):
        image -= np.array((123, 117, 104),np.float32)
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

# 作图工具
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

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
def draw_image_with_label_percent(image, label):
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    img_h, img_w = image.size[0], image.size[1]

    for item in label:
        # 边界框坐标（百分比 -> 像素）
        x0 = item[0] * img_w
        y0 = item[1] * img_h
        x1 = item[2] * img_w
        y1 = item[3] * img_h
        w, h = x1 - x0, y1 - y0

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
    data = CustomWiderFaceDataset(images_dir, images_label_path, img_size)
    print(len(data))
    img, label = data[2]
    print('==========================图片值：', img)
    print('==========================标签值：', label)
