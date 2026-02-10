# 🚗 AI-Driven Accident Reporting & FIR Automation System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An intelligent computer vision-based system that automates accident reporting and First Information Report (FIR) generation through real-time image analysis, vehicle detection, and damage assessment.

## 🎯 Key Features

- **Real-time Accident Detection**: YOLO-based object detection for vehicles and damage assessment
- **Automated FIR Generation**: Extract vehicle details, license plates, and damage severity automatically
- **OCR Integration**: Text extraction from license plates and vehicle documents with 90% accuracy
- **High Performance**: Process 1,000+ images/day with <2s inference latency
- **RESTful API**: FastAPI-based endpoints for easy integration
- **Dockerized Deployment**: Container-ready for seamless deployment

## 📊 Performance Metrics

- 🎯 **Detection Precision**: 88%
- 📝 **Text Extraction Accuracy**: 90%
- ⚡ **Inference Latency**: <2 seconds
- 📈 **Manual Entry Reduction**: 70%
- 🚀 **System Throughput Improvement**: 35%

## 🏗️ System Architecture

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│   Image     │─────▶│  Detection   │─────▶│  Damage     │
│   Input     │      │  Pipeline    │      │  Assessment │
└─────────────┘      └──────────────┘      └─────────────┘
                            │                      │
                            ▼                      ▼
                     ┌──────────────┐      ┌─────────────┐
                     │     OCR      │─────▶│     FIR     │
                     │  Extraction  │      │  Generation │
                     └──────────────┘      └─────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Docker (optional)
- CUDA-capable GPU (recommended for faster inference)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Saivishnu4/accident-fir-automation.git
cd accident-fir-automation
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download pre-trained models**
```bash
python scripts/download_models.py
```

5. **Run the application**
```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Docker Deployment

```bash
docker build -t accident-fir-system .
docker run -p 8000:8000 accident-fir-system
```

## 📁 Project Structure

```
accident-fir-automation/
│
├── src/
│   ├── api/                    # FastAPI application
│   ├── models/                 # ML model implementations
│   ├── preprocessing/          # Image preprocessing
│   ├── postprocessing/         # Result processing
│   └── utils/                  # Utility functions
│
├── models/                     # Trained model weights
├── data/                       # Dataset directory
├── notebooks/                  # Jupyter notebooks
├── tests/                      # Unit tests
├── scripts/                    # Utility scripts
├── docker/                     # Docker configurations
├── docs/                       # Documentation
└── requirements.txt
```

## 🔧 Usage

### API Endpoints

#### 1. Analyze Accident Image
```bash
curl -X POST "http://localhost:8000/api/v1/analyze" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@accident_image.jpg"
```

#### 2. Generate FIR Report
```bash
curl -X POST "http://localhost:8000/api/v1/fir/generate" \
  -H "Content-Type: application/json" \
  -d '{"analysis_id": "abc123"}'
```

### Python SDK

```python
from src.client import AccidentAnalyzer

analyzer = AccidentAnalyzer(api_url="http://localhost:8000")
result = analyzer.analyze_image("path/to/accident.jpg")
fir = analyzer.generate_fir(result['analysis_id'])
```

## 🧠 Model Details

- **YOLO Object Detection**: YOLOv8 for vehicle and damage detection (88% precision)
- **OCR Engine**: PaddleOCR/EasyOCR for text extraction (90% accuracy)
- **Damage Classification**: ResNet50-based classifier (87% accuracy)

## 📈 Training

```bash
# Train YOLO model
python scripts/train.py --model yolo --epochs 100

# Train damage classifier
python scripts/train.py --model damage --epochs 50
```

## 🧪 Testing

```bash
pytest tests/ -v --cov=src
```

## 📝 API Documentation

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 👥 Authors

**M SAIVISHNU SARODEame** - [GitHub](https://github.com/Saivishnu4)

---

⭐ If you find this project useful, please consider giving it a star!
