from src.test.tester import Tester

if __name__ == '__main__':
    cfg_tester = {
        'model_path': r'D:\Code\DL\Pytorch\retinaface\weights\model_best_20250518_021914.pth',
        'input_image_size': [320, 320]
    }

    test = Tester(cfg_tester=cfg_tester)
    test.detect_single_image(
        image_input=r'D:\Code\DL\Pytorch\retinaface\src\images\3_Riot_Riot_3_26.jpg'
    )
    test.detect_single_image(
        image_input=r'D:\Code\DL\Pytorch\retinaface\src\images\test.png'
    )
