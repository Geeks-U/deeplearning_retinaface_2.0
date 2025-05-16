import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from pathlib import Path
from datetime import datetime

from src.nets.retinaface import Retinaface
from src.data.dataset import CustomDataset, detection_collate
from src.utils.anchor import CustomAnchors  # 锚框生成器

from src.utils.loss import CustomLoss

cfg_anchor = {
    'input_image_size': [320, 320],
    'num_fpn_feature_layers': 3,
    'backbone_fpn_strides': [8, 16, 32],
    'num_anchor_per_pixel': 2,
    'anchor_ratios_per_level': [[8, 16], [32, 64], [128, 256]],
    'clip': False,
    'variance': [0.1, 0.2]
}

def save_model(model, path: Path):
    torch.save(model.state_dict(), path)
    print(f"Saved model to {path}")

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    num_epochs = 10
    batch_size = 16
    learning_rate = 1e-2

    # corner坐标，单位percent，范围0-1
    train_dataset = CustomDataset()
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, collate_fn=detection_collate
    )

    model = Retinaface().to(device)

    weights_save_dir = Path(r'D:\Code\DL\Pytorch\retinaface\weights')
    weights_save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    anchors = CustomAnchors(cfg_anchor=cfg_anchor).get_center_anchors().to(device)
    print("锚框形状: ", anchors.size())

    criterion = CustomLoss(num_classes=2, overlap_thresh=0.35, neg_pos=7, variance=[0.1, 0.2])

    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=5e-4
    )

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        total_loss, total_loc, total_conf, total_landm = 0, 0, 0, 0

        for i, (images, targets) in enumerate(train_loader):
            images = torch.from_numpy(images).float().to(device)
            targets = [torch.from_numpy(ann).float().to(device) for ann in targets]

            optimizer.zero_grad()

            out = model(images)

            loss_l, loss_c, loss_landm = criterion([out['bbox'], out['cls'], out['ldm']], anchors, targets)
            loss = 2 * loss_l + loss_c + loss_landm

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

        # 保存模型
        last_path = weights_save_dir / f'model_last_{timestamp}.pth'
        save_model(model, last_path)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = weights_save_dir / f'model_best_{timestamp}.pth'
            save_model(model, best_path)
