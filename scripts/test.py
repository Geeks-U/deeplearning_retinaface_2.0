import os
from src.test.tester import Tester

if __name__ == '__main__':
    # 当前脚本文件的绝对路径
    current_file = os.path.abspath(__file__)
    # 回到项目根目录 retinaface （假设脚本在 scripts 目录下）
    base_dir = os.path.dirname(os.path.dirname(current_file))

    model_path = os.path.join(base_dir, 'weights', 'model_best_20250518_021914.pth')
    image1 = os.path.join(base_dir, 'src', 'images', '3_Riot_Riot_3_26.jpg')
    image2 = os.path.join(base_dir, 'src', 'images', 'test.png')

    cfg_tester = {
        'model_path': model_path,
        'input_image_size': [320, 320]
    }

    test = Tester(cfg_tester=cfg_tester)
    test.detect_single_image(image_input=image1)
    test.detect_single_image(image_input=image2)
