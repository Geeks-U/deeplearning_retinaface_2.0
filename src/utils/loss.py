import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.anchor import match_center_anchor_to_gt_box_percent

def log_sum_exp(x):
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max

cfg_loss_default = {
    'num_classes': 2,
    'overlap_thresh': 0.35,
    'neg_pos': 7,
    'variance': [0.1, 0.2],
    'cuda': True
}

class CustomLoss(nn.Module):
    def __init__(self, cfg_loss=None):
        super().__init__()
        self.cfg = copy.deepcopy(cfg_loss_default)
        if cfg_loss is not None:
            self.cfg.update(cfg_loss)

        self.num_classes    = self.cfg['num_classes']
        self.threshold      = self.cfg['overlap_thresh']
        self.negpos_ratio   = self.cfg['neg_pos']
        self.variance       = self.cfg['variance']
        self.cuda           = self.cfg['cuda']

    def forward(self, predictions, priors, targets):
        #--------------------------------------------------------------------#
        #   取出预测结果的三个值：框的回归信息，置信度，人脸关键点的回归信息
        #--------------------------------------------------------------------#
        loc_data, conf_data, landm_data = predictions
        #--------------------------------------------------#
        #   计算出batch_size和先验框的数量
        #--------------------------------------------------#
        num         = loc_data.size(0)
        num_priors  = (priors.size(0))

        #--------------------------------------------------#
        #   创建一个tensor进行处理
        #--------------------------------------------------#
        loc_t   = torch.Tensor(num, num_priors, 4)
        landm_t = torch.Tensor(num, num_priors, 10)
        conf_t  = torch.LongTensor(num, num_priors)

        for idx in range(num):
            # 获得真实框与标签
            truths = targets[idx][:, :4].data
            labels = targets[idx][:, -1].data
            landms = targets[idx][:, 4:14].data

            # 获得先验框 张量的原始数据。priors.data 返回一个不包含梯度信息的张量视图
            defaults = priors.data
            #   利用真实框和先验框进行匹配。
            loc_t[idx], landm_t[idx], conf_t[idx] = match_center_anchor_to_gt_box_percent(defaults, truths, landms, labels,
                                                  self.threshold, self.variance)

        #--------------------------------------------------#
        #   转化成Variable
        #   loc_t   (num, num_priors, 4)
        #   conf_t  (num, num_priors)
        #   landm_t (num, num_priors, 10)
        #--------------------------------------------------#
        zeros = torch.tensor(0)
        if self.cuda:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
            landm_t = landm_t.cuda()
            zeros = zeros.cuda()

        # label为1表示人脸且含有关键点，-1表示人脸且没有关键点(在dataset中初始化)
        # 0表示背景板(在match_center_anchor_to_gt_box_percent中初始化)

        # 计算人脸框的loss时，就挑出含有人脸(label!=0)的数据，然后进行损失计算
        pos = conf_t != zeros
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        # 计算关键点loss时，就挑出含有关键点(label=1)的数据，然后进行损失计算
        pos1 = conf_t > zeros
        pos_idx1 = pos1.unsqueeze(pos1.dim()).expand_as(landm_data)
        landm_p = landm_data[pos_idx1].view(-1, 10)
        landm_t = landm_t[pos_idx1].view(-1, 10)
        loss_landm = F.smooth_l1_loss(landm_p, landm_t, reduction='sum')

        #--------------------------------------------------#
        #   batch_conf  (num * num_priors, 2)
        #   loss_c      (num, num_priors)
        #   conf_t      (num * num_priors, 1)
        #--------------------------------------------------#
        # 含有人脸就筛选出来
        conf_t[pos] = 1
        batch_conf = conf_data.view(-1, self.num_classes)
        # 这个地方是在寻找难分类的先验框 计算所有预测框和先验框的loss
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # 正样本(iou>threshold)意味着框里有人脸所以不算是难分类
        # 难分类的先验框不把正样本考虑进去，只考虑难分类的负样本，即label为0的样本
        # 就是被认作是背景的样本 这一步是为了增加一些难分类的样本来学习
        # 通过loss找到最难分类的前n个样本加入训练，n = self.negpos_ratio * 正样本数量
        # 目标是对于负样本要尽量学习 正样本要增加准确度
        loss_c[pos.view(-1, 1)] = 0
        loss_c = loss_c.view(num, -1)
        #--------------------------------------------------#
        #   loss_idx    (num, num_priors)
        #   idx_rank    (num, num_priors)
        #   num_pos     (num, )
        #   neg         (num, num_priors)
        #--------------------------------------------------#
        # 生成困难样本掩码
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        #--------------------------------------------------#
        #   pos_idx   (num, num_priors, num_classes)
        #   neg_idx   (num, num_priors, num_classes)
        #--------------------------------------------------#
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)

        # 选取出用于训练的正样本与负样本，计算loss
        # 训练样本构造
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1,self.num_classes)
        # 根据训练样本位置筛选目标样本
        targets_weighted = conf_t[(pos+neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')

        N = max(num_pos.data.sum().float(), 1)
        # box基于正样本(label!=0)
        loss_l /= N
        # num_neg >> num_pos，如果使用和来归一化会使得cls_loss较小，正样本的cls学习困难
        loss_c /= N

        num_pos_landm = pos1.long().sum(1, keepdim=True)
        # 基于label=1的样本
        N1 = max(num_pos_landm.data.sum().float(), 1)
        loss_landm /= N1
        return loss_l, loss_c, loss_landm
