# download_quick_start.py
from ultralytics import YOLO
import tensorflow as tf

# 1. YOLO (auto-downloads)
print("Downloading YOLO model...")
model = YOLO('yolov8m.pt')
model.save('models/yolov8_accident.pt')

# 2. ResNet (auto-downloads)
print("Downloading ResNet50...")
resnet = tf.keras.applications.ResNet50(
    weights='imagenet',
    include_top=False
)
resnet.save('models/resnet50_base.h5')

# 3. OCR (auto-downloads)
print("Downloading OCR models...")
from paddleocr import PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

print("All models downloaded successfully!")
