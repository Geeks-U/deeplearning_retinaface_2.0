import cv2
from src.test.retinaface import Retinaface

def run_single_image_predict(cfg_test):
    # 假设 Retinaface 类支持传入权重文件路径
    retinaface = Retinaface(cfg_test=cfg_test)

    image = cv2.imread(cfg_test['cfg_data']['img_path'])
    if image is None:
        print(f"Open Error! Failed to read image at: {cfg_test['cfg_data']['img_path']}")
        return

    # BGR to RGB for Retinaface
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Perform detection
    result_rgb = retinaface.detect_image(image_rgb)
    # RGB back to BGR for OpenCV display
    result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)

    cv2.imshow("Detection Result", result_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
