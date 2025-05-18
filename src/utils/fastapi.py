from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, FileResponse
from io import BytesIO
from PIL import Image
import torch
import uvicorn
import cv2
import os

from src.test.tester import Tester

app = FastAPI()

cfg_tester = {
    'model_path': r'D:\Code\DL\Pytorch\retinaface\weights\model_best_20250518_021914.pth',
    'input_image_size': [320, 320]
}
test = Tester(cfg_tester=cfg_tester)

@app.get("/")
async def get_html():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    html_path = os.path.join(current_dir, "fastapi.html")
    return FileResponse(html_path)

@app.post("/detect")
async def detect(frame: UploadFile = File(...)):
    image = Image.open(frame.file).convert('RGB')
    with torch.no_grad():
        detected_image = test.detect_single_image(image_input=image, return_image=True)

    image_bgr = cv2.cvtColor(detected_image, cv2.COLOR_RGB2BGR)

    success, buffer = cv2.imencode('.jpg', image_bgr)
    if not success:
        return {"error": "Image encoding failed"}

    return StreamingResponse(BytesIO(buffer.tobytes()), media_type="image/jpeg")

if __name__ == "__main__":
    print("模型路径：", cfg_tester['model_path'])
    print("点击打开视频检测demo http://localhost:8000")
    uvicorn.run(
        "src.utils.fastapi:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="warning"  # 只显示 warning 以上日志，屏蔽请求访问日志
    )
