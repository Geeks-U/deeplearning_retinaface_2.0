import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from src.nets.retinaface import Retinaface
from src.data.dataset import CustomDataset, detection_collate
from src.data.ds import CustomWiderFaceDataset
from src.utils.anchor import CustomAnchors  # 锚框生成器

# 自定义损失函数和工具函数（需要你实现）
from src.utils.loss import CustomLoss

cfg_anchor = {
        'input_image_size': [320, 320],
        'num_fpn_feature_layers': 3,
        'backbone_fpn_strides': [8, 16, 32],
        'num_anchor_per_pixel': 2,
        'anchor_ratios_per_level': [[8, 16], [32, 64], [128, 256]],
        'clip': False,
        # xy, wh的缩放倍数
        'variance': [0.1, 0.2]
    }

if __name__ == '__main__':
    # 训练设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 超参数设置
    num_epochs = 5
    batch_size = 16
    learning_rate = 1e-2  # SGD 通常使用较大的初始学习率

    # 数据加载
    train_dataset = CustomDataset()
    # train_dataset = CustomWiderFaceDataset()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=detection_collate)

    # 模型加载
    model = Retinaface().to(device)

    # 预制锚框生成
    anchors = CustomAnchors(cfg_anchor=cfg_anchor).get_center_anchors().to(device)
    print("锚框形状: ", anchors.size())

    # 损失函数
    criterion = CustomLoss(num_classes=2, overlap_thresh=0.35, neg_pos=7, variance=[0.1, 0.2])

    # 使用 SGD 优化器
    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=5e-4
    )

    # 学习率调度器（每10轮衰减一次）
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # 训练主循环
    for epoch in range(num_epochs):
        model.train()
        total_loss, total_loc, total_conf, total_landm = 0, 0, 0, 0

        for i, (images, targets) in enumerate(train_loader):
            # 转换为tensor
            images = torch.from_numpy(images).float().to(device)
            targets = [torch.from_numpy(ann).float().to(device) for ann in targets]
            if torch.isnan(images).any():
                print("图像数据中存在 NaN！")
            if torch.isinf(images).any():
                print("图像数据中存在 Inf！")

            optimizer.zero_grad()

            # 前向传播
            out = model(images)
            if torch.isnan(out['bbox']).any():
                print("bbox数据中存在 NaN！")
            if torch.isinf(out['bbox']).any():
                print("bbox数据中存在 Inf！")

            # 计算损失（定位、分类、关键点）
            loss_l, loss_c, loss_landm = criterion([out['bbox'], out['cls'], out['ldm']], anchors, targets)
            loss = 2 * loss_l + loss_c + loss_landm

            # 反向传播 + 优化
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_loc += loss_l.item()
            total_conf += loss_c.item()
            total_landm += loss_landm.item()

            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}, Loc: {loss_l.item():.4f}, "
                      f"Conf: {loss_c.item():.4f}, Landm: {loss_landm.item():.4f}")

        scheduler.step()

        avg_loss = total_loss / len(train_loader)
        avg_loc = total_loc / len(train_loader)
        avg_conf = total_conf / len(train_loader)
        avg_landm = total_landm / len(train_loader)

        print(f"[Epoch {epoch + 1}] Avg Loss: {avg_loss:.4f} "
              f"(Loc: {avg_loc:.4f}, Conf: {avg_conf:.4f}, Landm: {avg_landm:.4f})")

