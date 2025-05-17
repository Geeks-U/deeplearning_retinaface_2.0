from src.test.tester import Tester

if __name__ == '__main__':
    cfg_tester = {
        'model_path': r'D:\Code\DL\Pytorch\retinaface\weights\model_best_20250517_235034.pth',
        'input_image_size': [960, 960]
    }

    test = Tester(cfg_tester=cfg_tester)
    test.detect_single_image(
        image_path=r'D:\Code\DL\Pytorch\retinaface\src\images\29_Students_Schoolkids_Students_Schoolkids_29_60.jpg'
    )
