import time

import cv2
import numpy as np
import torch
import torch.nn as nn

from src.nets.retinaface import Retinaface as retin
from src.utils.anchor import CustomAnchors
from src.utils.utils import letterbox_image, preprocess_input
from src.utils.utils_box import (decode, decode_landm, non_max_suppression,
                              retinaface_correct_boxes)

class Retinaface(object):
    def __init__(self, cfg_test):
        # 配置加载
        self.cfg_test = cfg_test
        # 模型权重文件路径
        self.model_path = cfg_test['cfg_data']['model_path']
        self.cuda = cfg_test['cfg_hyperparameter']['CUDA']
        if self.cuda:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                print('CUDA is not available!')
                self.device = torch.device('cpu')
        else:
            self.device = torch.device('cpu')

        #---------------------------------------------------#
        #   先验框的生成
        #---------------------------------------------------#
        if cfg_test['cfg_hyperparameter']['letterbox_image']:
            self.anchors = CustomAnchors(cfg_anchor=cfg_test['cfg_anchor']).get_center_anchors()
        self.generate()

    #---------------------------------------------------#
    #   载入模型
    #---------------------------------------------------#
    def generate(self):
        #-------------------------------#
        #   载入模型与权值
        #-------------------------------#
        self.net    = retin().eval()
        self.net.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.net    = self.net.eval()
        print('{} model, and classes loaded.'.format(self.model_path))

        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image):
        #---------------------------------------------------#
        #   对输入图像进行一个备份，后面用于绘图
        #---------------------------------------------------#
        old_image = image.copy()
        #---------------------------------------------------#
        #   把图像转换成numpy的形式
        #---------------------------------------------------#
        image = np.array(image,np.float32)
        #---------------------------------------------------#
        #   计算输入图片的高和宽
        #---------------------------------------------------#
        im_height, im_width, _ = np.shape(image)
        #---------------------------------------------------#
        #   计算scale，用于将获得的预测框转换成原图的高宽
        #---------------------------------------------------#
        scale = [
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0]
        ]
        scale_for_landmarks = [
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
            np.shape(image)[1], np.shape(image)[0]
        ]

        #---------------------------------------------------------#
        #   letterbox_image可以给图像增加灰条，实现不失真的resize
        #---------------------------------------------------------#
        if self.cfg_test['cfg_hyperparameter']['letterbox_image']:
            image = letterbox_image(image, self.cfg_test['cfg_data']['image_size'])
        else:
            self.anchors = CustomAnchors(cfg_anchor=self.cfg_test['cfg_anchor']).get_center_anchors()

        with torch.no_grad():
            #-----------------------------------------------------------#
            #   图片预处理，归一化。
            #-----------------------------------------------------------#
            image = torch.from_numpy(preprocess_input(image).transpose(2, 0, 1)).unsqueeze(0).type(torch.FloatTensor)

            if self.cuda:
                self.anchors = self.anchors.cuda()
                image        = image.cuda()

            #---------------------------------------------------------#
            #   传入网络进行预测
            #---------------------------------------------------------#
            loc, conf, landms = self.net(image)

            #-----------------------------------------------------------#
            #   对预测框进行解码
            #-----------------------------------------------------------#
            boxes   = decode(loc.data.squeeze(0), self.anchors, self.cfg_test['cfg_anchor']['variance'])
            #-----------------------------------------------------------#
            #   获得预测结果的置信度
            #-----------------------------------------------------------#
            conf    = conf.data.squeeze(0)[:, 1:2]
            #-----------------------------------------------------------#
            #   对人脸关键点进行解码
            #-----------------------------------------------------------#
            landms  = decode_landm(landms.data.squeeze(0), self.anchors, self.cfg_test['cfg_anchor']['variance'])

            #-----------------------------------------------------------#
            #   对人脸识别结果进行堆叠 非极大抑制
            #-----------------------------------------------------------#
            boxes_conf_landms = torch.cat([boxes, conf, landms], -1)
            boxes_conf_landms = non_max_suppression(boxes_conf_landms, self.cfg_test['cfg_hyperparameter']['confidence'])

            if len(boxes_conf_landms) <= 0:
                return old_image

            #---------------------------------------------------------#
            #   如果使用了letterbox_image的话，要把灰条的部分去除掉。
            #---------------------------------------------------------#
            if self.cfg_test['cfg_hyperparameter']['letterbox_image']:
                boxes_conf_landms = retinaface_correct_boxes(boxes_conf_landms, \
                    np.array(self.cfg_test['cfg_data']['image_size']), np.array([im_height, im_width]))

        boxes_conf_landms[:, :4] = boxes_conf_landms[:, :4] * scale
        boxes_conf_landms[:, 5:] = boxes_conf_landms[:, 5:] * scale_for_landmarks

        for b in boxes_conf_landms:
            text = "{:.4f}".format(b[4])
            b = list(map(int, b))
            #---------------------------------------------------#
            #   b[0]-b[3]为人脸框的坐标，b[4]为得分
            #---------------------------------------------------#
            cv2.rectangle(old_image, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cx = b[0]
            cy = b[1] + 12
            cv2.putText(old_image, text, (cx, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

            print(b[0], b[1], b[2], b[3], b[4])
            #---------------------------------------------------#
            #   b[5]-b[14]为人脸关键点的坐标
            #---------------------------------------------------#
            cv2.circle(old_image, (b[5], b[6]), 1, (0, 0, 255), 4)
            cv2.circle(old_image, (b[7], b[8]), 1, (0, 255, 255), 4)
            cv2.circle(old_image, (b[9], b[10]), 1, (255, 0, 255), 4)
            cv2.circle(old_image, (b[11], b[12]), 1, (0, 255, 0), 4)
            cv2.circle(old_image, (b[13], b[14]), 1, (255, 0, 0), 4)
        return old_image
