import copy
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.nets.retinaface import Retinaface
from src.utils.anchor import CustomAnchors, calc_raw_bbox, calc_raw_landm, non_max_suppression


cfg_test_default = {
    'model_path': r'D:\Code\DL\Pytorch\retinaface\weights\model_best_20250517_184350.pth',
    'input_image_size': [960, 960],
    'cuda': True,
    'variance': [0.1, 0.2],
    'confidence': 0.5
}


class Tester:
    def __init__(self, cfg_tester=None):
        self.cfg = copy.deepcopy(cfg_test_default)
        if cfg_tester is not None:
            self.cfg.update(cfg_tester)

        self.device = 'cuda' if self.cfg['cuda'] and torch.cuda.is_available() else 'cpu'
        if self.cfg['cuda'] and self.device == 'cpu':
            print('CUDA is not available.')

        self.anchors = CustomAnchors(
            cfg_anchor={'input_image_size': self.cfg['input_image_size']}
        ).get_center_anchors().to(self.device)

        self.model = Retinaface()
        self.model.load_state_dict(torch.load(self.cfg['model_path']))
        self.model.to(self.device)
        self.model.eval()

    def detect_single_image(self, image_input, return_image: bool = False):
        if isinstance(image_input, str):
            image = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, Image.Image):
            image = image_input.convert('RGB')
        else:
            raise TypeError("image_input must be a file path or PIL.Image.Image")

        old_image = np.array(image).copy()
        img_w, img_h = image.size

        with torch.no_grad():
            image_resized = image.resize(self.cfg['input_image_size'], Image.BICUBIC)
            image_np = np.array(image_resized, dtype=np.float32)

            image_tensor = torch.from_numpy(
                (image_np - np.array([123.0, 117.0, 104.0], dtype=np.float32)).transpose(2, 0, 1)
            ).unsqueeze(0).float().to(self.device)

            outputs = self.model(image_tensor)
            bbox = calc_raw_bbox(outputs['bbox'].squeeze(0), self.anchors, self.cfg['variance'])
            cls = F.softmax(outputs['cls'], dim=-1).squeeze(0)[:, 1:2]
            ldm = calc_raw_landm(outputs['ldm'].squeeze(0), self.anchors, self.cfg['variance'])

            bbox_cls_ldm = torch.cat([bbox, cls, ldm], dim=-1)
            bbox_cls_ldm = non_max_suppression(bbox_cls_ldm, self.cfg['confidence'])
            if len(bbox_cls_ldm) <= 0:
                return old_image if return_image else None

        bbox_cls_ldm[:, :4] *= ([img_w, img_h] * 2)
        bbox_cls_ldm[:, 5:] *= ([img_w, img_h] * 5)

        for b in bbox_cls_ldm:
            text = f"{b[4]:.4f}"
            b = list(map(int, b))

            cv2.rectangle(old_image, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cv2.putText(old_image, text, (b[0], b[1] + 12), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

            colors = [(0, 0, 255), (0, 255, 255), (255, 0, 255), (0, 255, 0), (255, 0, 0)]
            for i, color in enumerate(colors):
                cv2.circle(old_image, (b[5 + i * 2], b[5 + i * 2 + 1]), 1, color, 4)

        image_bgr = cv2.cvtColor(old_image, cv2.COLOR_RGB2BGR)

        if return_image:
            return image_bgr
        else:
            cv2.imshow("Detection Result", image_bgr)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return None



if __name__ == '__main__':
    test = Tester()
    test.detect_single_image(
        image_path=r'D:\Code\DL\Pytorch\retinaface\src\images\29_Students_Schoolkids_Students_Schoolkids_29_60.jpg'
    )
