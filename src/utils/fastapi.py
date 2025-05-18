from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware  # 导入中间件
from io import BytesIO
from PIL import Image
import torch
import uvicorn
import numpy as np
import cv2

from src.test.tester import Tester

app = FastAPI()

# 允许跨域的域名列表，这里先开放所有，部署时可指定具体域名
origins = [
    "*"
]

# 添加中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,       # 允许所有来源跨域
    allow_credentials=True,
    allow_methods=["*"],         # 允许所有方法
    allow_headers=["*"],         # 允许所有请求头
)

cfg_tester = {
    'model_path': r'D:\Code\DL\Pytorch\retinaface\weights\model_best_20250518_021914.pth',
    'input_image_size': [320, 320]
}
test = Tester(cfg_tester=cfg_tester)

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
    uvicorn.run(
        "src.utils.fastapi:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="warning"  # 只显示 warning 以上日志，屏蔽请求访问日志
    )
