import copy
from math import ceil               # 向上取整函数
from itertools import product       # 用于生成笛卡尔积

import torch

cfg_anchor_default = {
    'input_image_size': [320, 320],
    'num_fpn_feature_layers': 3,
    'backbone_fpn_strides': [8, 16, 32],
    'num_anchor_per_pixel': 2,
    'anchor_ratios_per_level': [[8, 16], [32, 64], [128, 256]],
    'clip': True
}

class CustomAnchors(object):
    def __init__(self, cfg_anchor=None):
        super().__init__()
        self.cfg = copy.deepcopy(cfg_anchor_default)
        if cfg_anchor is not None:
            self.cfg.update(cfg_anchor)

        self.input_image_size = self.cfg['input_image_size']
        self.num_fpn_feature_layers = self.cfg['num_fpn_feature_layers']
        self.backbone_fpn_strides = self.cfg['backbone_fpn_strides']
        self.num_anchor_per_pixel = self.cfg['num_anchor_per_pixel']
        self.anchor_ratios_per_level = self.cfg['anchor_ratios_per_level']
        self.clip = self.cfg['clip']

        # 计算每个特征图的尺寸
        self.fpn_feature_output_size = [
            [
                ceil(self.input_image_size[0]/stride),
                ceil(self.input_image_size[1]/stride)
            ]
            for stride in self.backbone_fpn_strides
        ]

    # 中心锚框生成 x, y, w, h = 锚框中心坐标(x, y); 锚框宽高(w, h). 单位均为百分比
    def get_center_anchors(self):
        result = []
        for index, item in enumerate(self.fpn_feature_output_size):
            # 第index层特征图的锚框宽高变换比例
            anchor_ratios = self.anchor_ratios_per_level[index]
            # 第index层特征图的放缩比例
            stride = self.backbone_fpn_strides[index]
            # 遍历第index层特征图的每个像素点
            for h, w in product(range(item[0]), range(item[1])):
                # 为每个像素点生成宽高比例不同的锚框
                for anchor_ratio in anchor_ratios:
                    # 高度 宽度 百分比计算
                    anchor_h = anchor_ratio / self.input_image_size[0]
                    anchor_w = anchor_ratio / self.input_image_size[1]
                    # x坐标 y坐标 百分比计算(+0.5移位至每个像素的中心)
                    anchor_center_x = float((w + 0.5) * stride / self.input_image_size[1])
                    anchor_center_y = float((h + 0.5) * stride / self.input_image_size[0])

                    result.append([
                        anchor_center_x,
                        anchor_center_y,
                        anchor_w, anchor_h
                    ])
        result = torch.Tensor(result).view(-1, 4)
        # 将锚框约束到图片中(针对有可能超出图片边缘的锚框)
        if self.clip:
            # 先把 x_center ± w/2，y_center ± h/2 算出来，再限制在 [0,1]，最后恢复到中心坐标形式
            x_center, y_center, w, h = result[:, 0], result[:, 1], result[:, 2], result[:, 3]
            x1 = torch.clamp(x_center - w / 2, 0.0, 1.0)
            y1 = torch.clamp(y_center - h / 2, 0.0, 1.0)
            x2 = torch.clamp(x_center + w / 2, 0.0, 1.0)
            y2 = torch.clamp(y_center + h / 2, 0.0, 1.0)

            # 新的中心和宽高
            new_x_center = (x1 + x2) / 2
            new_y_center = (y1 + y2) / 2
            new_w = x2 - x1
            new_h = y2 - y1

            result = torch.stack([new_x_center, new_y_center, new_w, new_h], dim=1)

        return result

def box_center_to_corner(center_box):
    top_left = center_box[:, :2] - center_box[:, 2:] / 2
    bottom_right = center_box[:, :2] + center_box[:, 2:] / 2
    return torch.cat([top_left, bottom_right], dim=1)

# 此处理解需要对tensor的广播机制较为熟悉
# 计算cornerbox的iou
def iou_corner_boxes(corner_box0, corner_box1):
    A = corner_box0.size(0)
    B = corner_box1.size(0)
    # 交集框的左上角 [A,B,2]
    inter_lt_xy = torch.max(corner_box0[:, :2].unsqueeze(1).expand(A, B, 2),
                            corner_box1[:, :2].unsqueeze(0).expand(A, B, 2))
    # 交集框的右下角 [A,B,2]
    inter_rb_xy = torch.min(corner_box0[:, 2:].unsqueeze(1).expand(A, B, 2),
                            corner_box1[:, 2:].unsqueeze(0).expand(A, B, 2))
    # 计算交集面积 [A,B]
    inter_xy = torch.clamp((inter_rb_xy - inter_lt_xy), min=0)
    inter_area = inter_xy[:, :, 0] * inter_xy[:, :, 1]

    # 计算各自box的面积 [A,B]
    area_a = ((corner_box0[:, 2]-corner_box0[:, 0]) *
              (corner_box0[:, 3]-corner_box0[:, 1])).unsqueeze(1).expand_as(inter_area)
    area_b = ((corner_box1[:, 2]-corner_box1[:, 0]) *
              (corner_box1[:, 3]-corner_box1[:, 1])).unsqueeze(0).expand_as(inter_area)

    # 计算并集面积 [A,B]
    union = area_a + area_b - inter_area

    # 计算交并比 [A,B]
    IoU = inter_area / union
    return IoU


# center_anchors为中心锚框，corner_box_t为边角锚框, 数据单位均为percent
# prioranchor和真实anchor之间的匹配，给每个prior都找一个真实anchor
def match_center_anchor_to_gt_box_percent(center_anchors, corner_box_t, landm_t, score_t, threshold=0.35, variances=[0.1, 0.2]):
    # (true, prior)iou矩阵计算, iou最大值为1
    ious_tp = iou_corner_boxes(corner_box_t, box_center_to_corner(center_anchors))

    # 给每个真实框寻找最佳的先验框 第2个维度方向
    # [len(corner_box_t),], [len(corner_box_t),]
    best_prior_iou, best_prior_index = ious_tp.max(1, keepdim=False)

    # 给每个先验框寻找最佳的真实框 第1个维度方向
    # [len(corner_box_t),], [len(corner_box_t),]
    best_truth_iou, best_truth_index = ious_tp.max(0, keepdim=False)

    # 因为存在可能真实框最好的先验框的iou也为0
    # 所以给每个真实框分配至少一个先验框(此处的策略是将真实框最好的先验框iou赋值为2, 其实只需要>1即可, 也能保持最好)
    best_truth_iou.index_fill_(dim=0, index=best_prior_index, value=2)
    # 对应的修改 best_truth_iou best_truth_index
    for i in range(best_prior_iou.size(0)):
        best_prior_iou[i] = 2   # 未实际使用
        best_truth_index[best_prior_index[i]] = i

    # 获取每个先验框对应的最好的真实框 [len(center_anchors),4]
    match_box_t = corner_box_t[best_truth_index]
    # 获取每个先验框对应的最好的真实框的标签 [len(center_anchors),1], 1表示有人脸并且有特征点，-1表示有人脸无特征点
    match_label_t = score_t[best_truth_index]
    # 获取每个先验框对应的最好的真实框的特征点 [len(center_anchors),10]
    match_landm_t = landm_t[best_truth_index]

    # 将iou<阈值的box设为背景, 0表示为背景
    match_label_t[best_truth_iou < threshold] = 0

    # 计算预测锚框bbox和真实锚框bbox的Δx, Δy, Δw, Δh, 网络输出的bbox的目标值便是这个Δ
    # match_box_t =  (Δ + 1) * prior_box
    # Δ = (match_box_t - prior_box) / prior_box, Δ就相当于一个伸缩比例
    box_target = calc_target_bbox(match_box_t, center_anchors, variances)
    # 计算预测锚框landm和真实锚框landm的Δ, 网络输出的landm的目标值便是这个Δ
    # prior对landm的初始预测就是prior的中心(prior_x, prior_y)
    # match_landm_t = (prior_x, prior_y) + (Δw, Δh) * (prior_w, prior_h)
    # (Δw, Δh) = (match_landm_t - (prior_x, prior_y)) / (prior_w, prior_h)
    landm_target = calc_target_landm(match_landm_t, center_anchors, variances)

    return box_target, landm_target, match_label_t

# Δ = (match_box_t - prior_box) / prior_box[2:]
# 参数为corner框, center框，构造bbox的target
def calc_target_bbox(truth_corner_box, prior_center_box, variances):
    # 中心编码
    delta_xy = ((truth_corner_box[:, :2] + truth_corner_box[:, 2:]) / 2 - prior_center_box[:, :2]) / (variances[0] * prior_center_box[:, 2:])
    # 宽高编码
    delta_wh = (truth_corner_box[:, 2:] - truth_corner_box[:, :2]) / prior_center_box[:, 2:]
    delta_wh = torch.log(delta_wh) / variances[1]
    return torch.cat([delta_xy, delta_wh], 1)

# (Δw, Δh) = (match_landm_t - (prior_x, prior_y)) / (prior_w, prior_h)
# 参数为center框，构造landm的target
def calc_target_landm(truth_landm, prior_center_box, variances):
    truth_landm = torch.reshape(truth_landm, (truth_landm.size(0), 5, 2))
    priors_cx = prior_center_box[:, 0].unsqueeze(1).expand(truth_landm.size(0), 5).unsqueeze(2)
    priors_cy = prior_center_box[:, 1].unsqueeze(1).expand(truth_landm.size(0), 5).unsqueeze(2)
    priors_w = prior_center_box[:, 2].unsqueeze(1).expand(truth_landm.size(0), 5).unsqueeze(2)
    priors_h = prior_center_box[:, 3].unsqueeze(1).expand(truth_landm.size(0), 5).unsqueeze(2)
    prior_center_box = torch.cat([priors_cx, priors_cy, priors_w, priors_h], dim=2)

    # 减去中心后除上宽高
    g_cxcy = truth_landm[:, :, :2] - prior_center_box[:, :, :2]
    g_cxcy /= (variances[0] * prior_center_box[:, :, 2:])
    g_cxcy = g_cxcy.reshape(g_cxcy.size(0), -1)
    return g_cxcy

# 测试代码 --------------------------------------------------
if __name__ == "__main__":
    # # 锚框生成测试
    # cfg_anchor = {
    #     'input_image_size': [960, 960],
    #     'num_fpn_feature_layers': 3,
    #     'backbone_fpn_strides': [8, 16, 32],
    #     'num_anchor_per_pixel': 2,
    #     'anchor_ratios_per_level': [[8, 16], [32, 64], [128, 256]],
    #     'clip': False,
    #     'variance': [0.1, 0.2]
    # }
    # res = CustomAnchors(cfg_anchor=cfg_anchor).get_center_anchors()
    # print('锚框形状: ', res.shape)

    # # box转换函数测试
    # print(box_center_to_corner(center_box=np.array([[50, 50, 20, 10], [30, 50, 10, 10]], dtype=np.float32)))

    # # iou_corner_boxes交并比计算测试
    # boxes1 = torch.tensor([[20, 10, 20, 40], [30, 50, 110, 110]], dtype=torch.float32)
    # boxes2 = torch.tensor([[50, 50, 120, 110], [3, 5, 10, 10]], dtype=torch.float32)
    # iou_values = iou_corner_boxes(boxes1, boxes2)
    # print(iou_values)

    # match_center_anchor_to_gt_box_percent匹配测试
    center_anchors = torch.tensor([[0.5, 0.5, 0.5, 0.5]], dtype=torch.float32)
    corner_box_t = torch.tensor([[0.25, 0.25, 0.75, 0.75]], dtype=torch.float32)
    res = match_center_anchor_to_gt_box_percent(center_anchors, corner_box_t)
    print(res)
