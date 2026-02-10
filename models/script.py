#!/bin/bash
# setup_models_and_data.sh

echo "Setting up models and datasets..."

# Create directories
mkdir -p models data/{raw,processed}/{vehicles,damage,plates}

# 1. Download YOLO weights
echo "Downloading YOLOv8 weights..."
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt -P models/
mv models/yolov8m.pt models/yolov8_accident.pt

# 2. Download datasets from Kaggle
echo "Installing Kaggle CLI..."
pip install kaggle

echo "Downloading vehicle damage dataset..."
kaggle datasets download -d anujms/car-damage-detection -p data/raw/
unzip data/raw/car-damage-detection.zip -d data/raw/damage/

echo "Downloading license plate dataset..."
kaggle datasets download -d dataturks/vehicle-number-plate-detection -p data/raw/
unzip data/raw/vehicle-number-plate-detection.zip -d data/raw/plates/

# 3. Download COCO subset (vehicles only)
echo "Downloading COCO vehicle images..."
python3 << EOF
import fiftyone as fo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="train",
    label_types=["detections"],
    classes=["car", "truck", "bus", "motorcycle", "bicycle"],
    max_samples=5000
)

dataset.export(
    export_dir="data/raw/vehicles",
    dataset_type=fo.types.YOLOv5Dataset,
    label_field="ground_truth"
)
EOF

# 4. Initialize OCR models (auto-download)
echo "Initializing OCR models..."
python3 << EOF
from paddleocr import PaddleOCR
import easyocr

# Downloads models automatically
ocr = PaddleOCR(use_angle_cls=True, lang='en')
reader = easyocr.Reader(['en'])
print("OCR models downloaded successfully!")
EOF

echo "Setup complete!"
echo "Models saved to: models/"
echo "Datasets saved to: data/raw/"
