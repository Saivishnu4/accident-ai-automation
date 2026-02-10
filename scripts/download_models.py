#!/usr/bin/env python3
"""
Automated Model Weights and Dataset Downloader
Downloads all necessary models and datasets for the Accident FIR Automation System
"""

import os
import sys
import argparse
import urllib.request
from pathlib import Path
import zipfile
import tarfile
import shutil
import subprocess


class Downloader:
    def __init__(self, base_dir="."):
        self.base_dir = Path(base_dir)
        self.models_dir = self.base_dir / "models"
        self.data_dir = self.base_dir / "data"
        
        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "raw").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "processed").mkdir(parents=True, exist_ok=True)
    
    def download_file(self, url, destination):
        """Download file with progress bar"""
        print(f"Downloading {url}...")
        
        def progress_hook(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            sys.stdout.write(f"\r...{percent}%")
            sys.stdout.flush()
        
        try:
            urllib.request.urlretrieve(url, destination, progress_hook)
            print("\n✓ Download complete")
            return True
        except Exception as e:
            print(f"\n✗ Download failed: {e}")
            return False
    
    def download_yolo_weights(self, model_size="m"):
        """Download YOLOv8 weights"""
        print("\n" + "="*60)
        print("Downloading YOLOv8 Model Weights")
        print("="*60)
        
        model_sizes = {
            "n": "yolov8n.pt",  # Nano - fastest
            "s": "yolov8s.pt",  # Small
            "m": "yolov8m.pt",  # Medium - recommended
            "l": "yolov8l.pt",  # Large
            "x": "yolov8x.pt",  # Extra Large - most accurate
        }
        
        if model_size not in model_sizes:
            print(f"Invalid model size. Choose from: {list(model_sizes.keys())}")
            return False
        
        model_name = model_sizes[model_size]
        url = f"https://github.com/ultralytics/assets/releases/download/v0.0.0/{model_name}"
        destination = self.models_dir / "yolov8_accident.pt"
        
        if destination.exists():
            print(f"Model already exists at {destination}")
            response = input("Download again? (y/n): ")
            if response.lower() != 'y':
                return True
        
        success = self.download_file(url, destination)
        if success:
            print(f"YOLOv8 model saved to: {destination}")
        return success
    
    def download_resnet_weights(self):
        """Download ResNet50 base weights"""
        print("\n" + "="*60)
        print("Downloading ResNet50 Weights (via Keras)")
        print("="*60)
        
        try:
            import tensorflow as tf
            from tensorflow.keras.applications import ResNet50
            
            print("Downloading ImageNet weights...")
            model = ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=(224, 224, 3)
            )
            
            save_path = self.models_dir / "resnet50_base.h5"
            model.save(save_path)
            print(f"✓ ResNet50 saved to: {save_path}")
            return True
            
        except ImportError:
            print("✗ TensorFlow not installed. Install with: pip install tensorflow")
            return False
        except Exception as e:
            print(f"✗ Failed to download ResNet50: {e}")
            return False
    
    def setup_ocr_models(self):
        """Initialize OCR models (they auto-download)"""
        print("\n" + "="*60)
        print("Setting up OCR Models")
        print("="*60)
        
        # PaddleOCR
        try:
            print("Initializing PaddleOCR...")
            from paddleocr import PaddleOCR
            ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
            print("✓ PaddleOCR models downloaded")
        except ImportError:
            print("⚠ PaddleOCR not installed. Install with: pip install paddleocr")
        except Exception as e:
            print(f"⚠ PaddleOCR setup issue: {e}")
        
        # EasyOCR
        try:
            print("Initializing EasyOCR...")
            import easyocr
            reader = easyocr.Reader(['en'], verbose=False)
            print("✓ EasyOCR models downloaded")
        except ImportError:
            print("⚠ EasyOCR not installed. Install with: pip install easyocr")
        except Exception as e:
            print(f"⚠ EasyOCR setup issue: {e}")
        
        return True
    
    def download_sample_dataset(self):
        """Download a sample dataset using fiftyone"""
        print("\n" + "="*60)
        print("Downloading Sample Dataset (COCO vehicles)")
        print("="*60)
        
        try:
            import fiftyone as fo
            import fiftyone.zoo as foz
            
            print("Downloading COCO dataset (vehicle classes only)...")
            print("This may take a while (downloading 1000 samples)...")
            
            dataset = foz.load_zoo_dataset(
                "coco-2017",
                split="train",
                label_types=["detections"],
                classes=["car", "truck", "bus", "motorcycle", "bicycle"],
                max_samples=1000,
                dataset_name="accident_vehicles"
            )
            
            # Export to YOLO format
            export_dir = self.data_dir / "raw" / "coco_vehicles"
            export_dir.mkdir(parents=True, exist_ok=True)
            
            dataset.export(
                export_dir=str(export_dir),
                dataset_type=fo.types.YOLOv5Dataset,
                label_field="ground_truth"
            )
            
            print(f"✓ Dataset exported to: {export_dir}")
            return True
            
        except ImportError:
            print("⚠ FiftyOne not installed. Install with: pip install fiftyone")
            print("Skipping dataset download...")
            return False
        except Exception as e:
            print(f"✗ Dataset download failed: {e}")
            return False
    
    def download_kaggle_dataset(self, dataset_id):
        """Download dataset from Kaggle"""
        print(f"\nDownloading Kaggle dataset: {dataset_id}")
        
        # Check if kaggle is installed
        try:
            import kaggle
        except ImportError:
            print("✗ Kaggle not installed. Install with: pip install kaggle")
            return False
        
        # Check if credentials exist
        kaggle_creds = Path.home() / ".kaggle" / "kaggle.json"
        if not kaggle_creds.exists():
            print("✗ Kaggle credentials not found!")
            print("Please follow these steps:")
            print("1. Go to https://www.kaggle.com/settings")
            print("2. Scroll to 'API' section")
            print("3. Click 'Create New API Token'")
            print("4. Move kaggle.json to ~/.kaggle/")
            return False
        
        try:
            output_dir = self.data_dir / "raw" / "kaggle"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            subprocess.run([
                "kaggle", "datasets", "download",
                "-d", dataset_id,
                "-p", str(output_dir),
                "--unzip"
            ], check=True)
            
            print(f"✓ Dataset downloaded to: {output_dir}")
            return True
            
        except Exception as e:
            print(f"✗ Download failed: {e}")
            return False
    
    def create_dataset_yaml(self):
        """Create YOLO dataset configuration"""
        yaml_content = """# Dataset configuration for YOLOv8
path: ../data/processed
train: train/images
val: val/images
test: test/images

# Classes
names:
  0: car
  1: motorcycle
  2: truck
  3: bus
  4: bicycle
  5: front_damage
  6: rear_damage
  7: side_damage
  8: broken_glass
  9: deployed_airbag

nc: 10  # number of classes
"""
        
        yaml_path = self.data_dir / "dataset.yaml"
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"✓ Dataset config created: {yaml_path}")
    
    def show_summary(self):
        """Show download summary"""
        print("\n" + "="*60)
        print("DOWNLOAD SUMMARY")
        print("="*60)
        
        print("\nModels directory:", self.models_dir)
        if self.models_dir.exists():
            models = list(self.models_dir.glob("*"))
            if models:
                for model in models:
                    size = model.stat().st_size / (1024 * 1024)  # MB
                    print(f"  ✓ {model.name} ({size:.1f} MB)")
            else:
                print("  (empty)")
        
        print("\nData directory:", self.data_dir)
        if (self.data_dir / "raw").exists():
            datasets = list((self.data_dir / "raw").iterdir())
            if datasets:
                for dataset in datasets:
                    if dataset.is_dir():
                        file_count = len(list(dataset.rglob("*")))
                        print(f"  ✓ {dataset.name} ({file_count} files)")
            else:
                print("  (empty)")
        
        print("\n" + "="*60)
        print("Next Steps:")
        print("="*60)
        print("1. Review downloaded models in:", self.models_dir)
        print("2. Prepare your dataset or use downloaded samples")
        print("3. Update .env file with model paths")
        print("4. Run training: python scripts/train.py")
        print("5. Start API: uvicorn src.api.main:app --reload")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Download models and datasets for Accident FIR Automation"
    )
    parser.add_argument(
        '--models-only',
        action='store_true',
        help='Download only model weights'
    )
    parser.add_argument(
        '--data-only',
        action='store_true',
        help='Download only datasets'
    )
    parser.add_argument(
        '--yolo-size',
        choices=['n', 's', 'm', 'l', 'x'],
        default='m',
        help='YOLO model size (n=nano, s=small, m=medium, l=large, x=xlarge)'
    )
    parser.add_argument(
        '--kaggle-dataset',
        type=str,
        help='Kaggle dataset ID to download (e.g., anujms/car-damage-detection)'
    )
    parser.add_argument(
        '--skip-ocr',
        action='store_true',
        help='Skip OCR model initialization'
    )
    parser.add_argument(
        '--skip-sample-data',
        action='store_true',
        help='Skip downloading sample COCO dataset'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("Accident FIR Automation - Model & Data Downloader")
    print("="*60)
    
    downloader = Downloader()
    
    # Download models
    if not args.data_only:
        downloader.download_yolo_weights(args.yolo_size)
        downloader.download_resnet_weights()
        if not args.skip_ocr:
            downloader.setup_ocr_models()
    
    # Download datasets
    if not args.models_only:
        if not args.skip_sample_data:
            downloader.download_sample_dataset()
        
        if args.kaggle_dataset:
            downloader.download_kaggle_dataset(args.kaggle_dataset)
        
        downloader.create_dataset_yaml()
    
    # Show summary
    downloader.show_summary()


if __name__ == "__main__":
    main()
