import sys
import torch
from models.common import DetectMultiBackend
from utils.general import non_max_suppression
from utils.augmentations import letterbox
import cv2
import numpy as np


def detect_image(image_path):
    # YOLO 모델 설정
    weights = './runs/train/exp17/weights/best.pt'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DetectMultiBackend(weights, device=device)
    imgsz = 640

    # 이미지 불러오기 및 전처리
    image = cv2.imread(image_path)
    image = letterbox(image, imgsz, stride=32, auto=True)[0]
    image = image.transpose((2, 0, 1))[::-1]  # BGR to RGB, HWC to CHW
    image = np.ascontiguousarray(image)

    # 추론
    image = torch.from_numpy(image).to(device).float() / 255.0
    if image.ndimension() == 3:
        image = image.unsqueeze(0)

    # 탐지 실행
    pred = model(image, augment=False, visualize=False)
    pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False, max_det=1000)

    # 바운딩 박스 그리기
    img = image.squeeze(0).permute(1, 2, 0).cpu().numpy()  # PyTorch 텐서에서 NumPy 이미지로 변환
    img = (img * 255).astype(np.uint8).copy()  # 스케일 조정

    # 객체 탐지 결과에 따른 메시지 출력
    detected = False
    for i, det in enumerate(pred):
        if len(det):
            for *xyxy, conf, cls in det:
                label = model.names[int(cls)]
                if label == 'empire':  # 'empire' 레이블이 있는 경우
                    detected = True
                    x1, y1, x2, y2 = map(int, xyxy)  # 좌표를 정수로 변환
                    print(f"True(empire state building이 맞습니다)")
                    # print(f"Bounding Box Coordinates: x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}")
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    break
            if detected:
                break
        if not detected:
            print("False(empire state building이 아닙니다)")
    cv2.imshow('Detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = sys.argv[1]  # 명령줄 인수에서 이미지 경로 받기
    detect_image(image_path)