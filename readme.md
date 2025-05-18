# RetinaFace Face Detection

基于PyTorch实现的RetinaFace人脸检测模型，支持人脸检测和关键点定位。

## 功能特点

- 人脸检测：支持多尺度人脸检测
- 关键点定位：支持5点关键点定位
- FastAPI服务：提供HTTP API接口，支持实时视频流检测
- 实时预览：提供Web界面实时预览检测结果

## 环境要求

- Python 3.8+
- PyTorch 2.7.0+
- CUDA 12.8+ (GPU加速)
- OpenCV 4.11.0+
- FastAPI 0.115.12+
- 其他依赖见 requirements.txt

## 安装

1. 克隆仓库：
```bash
git clone https://github.com/Geeks-U/deeplearning_retinaface_2.0.git
cd retinaface
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 训练模型

```bash
python -m scripts.train
```

训练配置可在 `scripts/train.py` 中修改：
- num_epochs: 训练轮数
- batch_size: 批次大小
- input_image_size: 输入图像尺寸

训练数据集配置可在 `src/data/dataset.py` 中修改：
- image_dir: 图片目录,
- label_path: 标签文件路径,

数据集下载链接: https://pan.baidu.com/s/1uwztX937jhSUlOsG3QHRqA?pwd=s1sh 提取码: s1sh

### 2. 测试模型

```bash
python -m scripts.test
```

测试配置可在 `scripts/test.py` 中修改：
- model_path: 模型权重路径
- input_image_size: 输入图像尺寸

### 3. 启动FastAPI服务

```bash
python -m src.utils.fastapi
```

服务将在 http://localhost:8000 启动，提供以下API：

- POST /detect：接收图片文件，返回检测结果
  - 请求：multipart/form-data 格式的图片文件
  - 响应：检测后的图片（JPEG格式）

### 4. 实时预览

访问 http://localhost:8000 可以打开实时预览界面：
- 左侧显示摄像头原始画面
- 右侧显示检测结果
- 支持实时帧率显示
- 支持采样频率调整

## 项目结构

```
retinaface/
├── config/             # 配置文件
├── scripts/            # 脚本文件
│   ├── train.py       # 训练脚本
│   └── test.py        # 测试脚本
├── src/               # 源代码
│   ├── models/        # 模型定义
│   ├── utils/         # 工具函数
│   │   └── fastapi.py # FastAPI服务
│   ├── train/         # 训练相关
│   └── test/          # 测试相关
├── weights/           # 模型权重
└── requirements.txt   # 项目依赖
```

## 注意事项

1. 首次训练需要下载训练数据集
2. GPU加速需要安装对应版本的CUDA和cuDNN
3. FastAPI服务默认监听所有网络接口，生产环境部署时请注意安全配置

## Reference
https://github.com/bubbliiiing/retinaface-pytorch.git