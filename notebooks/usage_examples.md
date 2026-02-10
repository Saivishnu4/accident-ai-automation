# Accident FIR Automation - Usage Examples

This notebook demonstrates how to use the Accident FIR Automation system.

## Setup

```python
import sys
sys.path.append('..')

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.models.yolo_detector import YOLODetector
from src.models.ocr_engine import OCREngine
from src.models.damage_classifier import DamageClassifier

%matplotlib inline
```

## 1. Vehicle Detection

```python
# Initialize YOLO detector
detector = YOLODetector(
    model_path='../models/yolov8_accident.pt',
    confidence_threshold=0.5
)

# Load test image
image_path = '../data/test/accident_001.jpg'
image = cv2.imread(image_path)

# Detect vehicles and damage
detections = detector.detect(image)

print(f"Total detections: {len(detections)}")
for det in detections:
    print(f"- {det['class']} (confidence: {det['confidence']:.2f})")
```

## 2. Visualize Detections

```python
# Annotate image
annotated = detector.annotate_image(image, detections)

# Display
plt.figure(figsize=(15, 10))
plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
plt.title('Vehicle and Damage Detection')
plt.axis('off')
plt.show()
```

## 3. License Plate Recognition

```python
# Initialize OCR engine
ocr = OCREngine(backend='paddle', use_gpu=False)

# Extract vehicle bounding boxes
vehicle_bboxes = [d['bbox'] for d in detections if d['type'] == 'vehicle']

# Extract license plates
license_plates = ocr.extract_license_plates(image, vehicle_bboxes)

print("Detected license plates:")
for plate in license_plates:
    print(f"- {plate['text']} (confidence: {plate['confidence']:.2f})")
```

## 4. Damage Assessment

```python
# Initialize damage classifier
classifier = DamageClassifier(
    model_path='../models/damage_classifier.h5'
)

# Get damage regions
damage_bboxes = [d['bbox'] for d in detections if d['type'] == 'damage']

# Classify each damage region
damage_assessments = classifier.classify_multiple(image, damage_bboxes)

print("Damage assessments:")
for i, assessment in enumerate(damage_assessments):
    print(f"Region {i+1}: {assessment['severity']} "
          f"(score: {assessment['confidence']:.2f})")
```

## 5. Complete Analysis Pipeline

```python
def analyze_accident_image(image_path):
    """Complete accident analysis pipeline"""
    
    # Load image
    image = cv2.imread(image_path)
    
    # 1. Detect vehicles and damage
    detections = detector.detect(image)
    vehicles = [d for d in detections if d['type'] == 'vehicle']
    damages = [d for d in detections if d['type'] == 'damage']
    
    # 2. Extract license plates
    vehicle_bboxes = [v['bbox'] for v in vehicles]
    plates = ocr.extract_license_plates(image, vehicle_bboxes)
    
    # 3. Assess damage severity
    damage_bboxes = [d['bbox'] for d in damages]
    damage_assessments = classifier.classify_multiple(image, damage_bboxes)
    overall_damage = classifier.aggregate_damage_assessment(damage_assessments)
    
    # 4. Compile results
    results = {
        'vehicles_detected': len(vehicles),
        'damage_indicators': len(damages),
        'license_plates': [p['text'] for p in plates],
        'overall_severity': overall_damage['overall_severity'],
        'severity_score': overall_damage['severity_score']
    }
    
    return results, detections

# Test on sample image
results, detections = analyze_accident_image('../data/test/accident_001.jpg')
print("Analysis Results:")
print(f"Vehicles: {results['vehicles_detected']}")
print(f"Damage indicators: {results['damage_indicators']}")
print(f"License plates: {', '.join(results['license_plates'])}")
print(f"Overall severity: {results['overall_severity']} "
      f"(score: {results['severity_score']:.2f})")
```

## 6. Batch Processing

```python
from glob import glob
import pandas as pd

def batch_analyze(image_dir):
    """Analyze multiple images"""
    
    image_paths = glob(f"{image_dir}/*.jpg")
    results_list = []
    
    for img_path in image_paths:
        try:
            results, _ = analyze_accident_image(img_path)
            results['image_path'] = img_path
            results_list.append(results)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    return pd.DataFrame(results_list)

# Batch process test images
df_results = batch_analyze('../data/test')
print(df_results)

# Summary statistics
print("\nSummary:")
print(f"Total images: {len(df_results)}")
print(f"Average severity score: {df_results['severity_score'].mean():.2f}")
print(f"Vehicles detected: {df_results['vehicles_detected'].sum()}")
```

## 7. Generate Report

```python
def generate_accident_report(results, detections, output_path='report.txt'):
    """Generate text report"""
    
    with open(output_path, 'w') as f:
        f.write("ACCIDENT ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Vehicles Detected: {results['vehicles_detected']}\n")
        f.write(f"Damage Indicators: {results['damage_indicators']}\n")
        f.write(f"Overall Severity: {results['overall_severity']}\n")
        f.write(f"Severity Score: {results['severity_score']:.2f}/1.00\n\n")
        
        f.write("License Plates:\n")
        for plate in results['license_plates']:
            f.write(f"  - {plate}\n")
        
        f.write("\nDetailed Detections:\n")
        for i, det in enumerate(detections):
            f.write(f"{i+1}. {det['class']} - "
                   f"Confidence: {det['confidence']:.2f}\n")
    
    print(f"Report saved to {output_path}")

# Generate report
results, detections = analyze_accident_image('../data/test/accident_001.jpg')
generate_accident_report(results, detections)
```

## 8. Using the API

```python
import requests

# API endpoint
API_URL = "http://localhost:8000"

# Upload and analyze image
def api_analyze_image(image_path):
    with open(image_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(
            f"{API_URL}/api/v1/analyze",
            files=files
        )
    return response.json()

# Example usage
# result = api_analyze_image('../data/test/accident_001.jpg')
# print(result)
```

## 9. Model Training Example

```python
# This demonstrates how to train custom models
# See scripts/train.py for complete training pipeline

from ultralytics import YOLO

# Initialize model
model = YOLO('yolov8n.pt')

# Train
results = model.train(
    data='../data/dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='accident_detector'
)

# Validate
metrics = model.val()
print(f"mAP50: {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")
```

## 10. Performance Benchmarking

```python
import time

def benchmark_inference(model, image, num_runs=100):
    """Benchmark model inference time"""
    
    times = []
    for _ in range(num_runs):
        start = time.time()
        _ = model.detect(image)
        times.append(time.time() - start)
    
    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times)
    }

# Benchmark
image = cv2.imread('../data/test/accident_001.jpg')
stats = benchmark_inference(detector, image)

print("Inference Time Statistics (seconds):")
print(f"Mean: {stats['mean']:.4f}")
print(f"Std Dev: {stats['std']:.4f}")
print(f"Min: {stats['min']:.4f}")
print(f"Max: {stats['max']:.4f}")
```
