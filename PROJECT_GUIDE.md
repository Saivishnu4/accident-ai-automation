# Accident FIR Automation - Complete Project Guide

## Table of Contents
1. [Project Overview](#project-overview)
2. [System Requirements](#system-requirements)
3. [Installation](#installation)
4. [Dataset Preparation](#dataset-preparation)
5. [Model Training](#model-training)
6. [Running the API](#running-the-api)
7. [Testing](#testing)
8. [Deployment](#deployment)
9. [Monitoring](#monitoring)
10. [Troubleshooting](#troubleshooting)

---

## Project Overview

This system automates accident reporting and FIR generation using:
- **YOLOv8** for vehicle and damage detection
- **PaddleOCR/EasyOCR** for license plate recognition
- **ResNet50** for damage severity classification
- **FastAPI** for RESTful API endpoints
- **Docker** for containerized deployment

### Key Metrics
- Detection Precision: 88%
- OCR Accuracy: 90%
- Processing Time: <2s per image
- Throughput: 1000+ images/day

---

## System Requirements

### Minimum Requirements
- Python 3.8+
- 8GB RAM
- 50GB storage
- CPU: 4 cores

### Recommended for Production
- Python 3.10+
- 16GB RAM
- 100GB SSD storage
- GPU: NVIDIA with 6GB+ VRAM (CUDA 11.8+)
- CPU: 8 cores

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/Saivishnu4/accident-fir-automation.git
cd accident-fir-automation
```

### 2. Set Up Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# For development
pip install -r requirements-dev.txt
```

### 4. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your settings
nano .env
```

### 5. Download Pre-trained Models

```bash
# Download YOLO weights
python scripts/download_models.py --model yolo

# Download damage classifier
python scripts/download_models.py --model damage

# Or download all models
python scripts/download_models.py --all
```

---

## Dataset Preparation

### Dataset Structure

```
data/
├── raw/
│   ├── accidents/
│   │   ├── accident_001.jpg
│   │   ├── accident_002.jpg
│   │   └── ...
│   ├── vehicles/
│   └── damage/
├── processed/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── val/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       ├── images/
│       └── labels/
└── annotations/
    └── dataset.yaml
```

### 1. Collect Images

Gather accident scene images with:
- Different vehicle types (cars, motorcycles, trucks, buses)
- Various damage levels (minor to severe)
- Clear license plates
- Different lighting and weather conditions

Recommended: 10,000+ images for training

### 2. Annotate Images

#### For YOLO (Object Detection):

Use [Labelimg](https://github.com/heartexlabs/labelImg) or [CVAT](https://github.com/opencv/cvat):

```bash
# Install labelImg
pip install labelImg

# Run annotation tool
labelImg data/raw/accidents/ data/annotations/
```

**Classes to annotate:**
- car
- motorcycle
- truck
- bus
- bicycle
- front_damage
- rear_damage
- side_damage
- broken_glass
- deployed_airbag

#### For Damage Classification:

Organize images by damage severity:

```
data/damage_classification/
├── none/
├── minor/
├── moderate/
├── severe/
└── total/
```

### 3. Create Dataset Configuration

Create `data/dataset.yaml`:

```yaml
path: ../data/processed
train: train/images
val: val/images
test: test/images

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
```

### 4. Split Dataset

```bash
python scripts/prepare_dataset.py \
  --input data/raw \
  --output data/processed \
  --train-split 0.7 \
  --val-split 0.2 \
  --test-split 0.1
```

---

## Model Training

### 1. Train YOLO Detector

```bash
python scripts/train.py \
  --model yolo \
  --data data/dataset.yaml \
  --epochs 100 \
  --batch-size 16 \
  --image-size 640 \
  --pretrained \
  --device 0  # GPU index, or 'cpu'
```

**Training arguments:**
- `--epochs`: Number of training epochs (100-300 recommended)
- `--batch-size`: Batch size (16-32 for GPU, 4-8 for CPU)
- `--image-size`: Input image size (640, 1280)
- `--pretrained`: Use pre-trained YOLOv8 weights
- `--patience`: Early stopping patience (default: 10)

**Expected training time:**
- GPU (RTX 3080): ~6-8 hours for 100 epochs
- CPU: ~48-72 hours for 100 epochs

### 2. Train Damage Classifier

```bash
python scripts/train.py \
  --model damage \
  --data-dir data/damage_classification \
  --epochs 50 \
  --batch-size 32 \
  --image-size 224 \
  --learning-rate 0.001
```

**Expected training time:**
- GPU: ~2-3 hours
- CPU: ~12-18 hours

### 3. Monitor Training

```bash
# View TensorBoard logs
tensorboard --logdir outputs/training/logs

# Access at http://localhost:6006
```

### 4. Evaluate Models

```bash
# Evaluate YOLO
python scripts/evaluate.py \
  --model yolo \
  --weights outputs/training/best.pt \
  --data data/dataset.yaml

# Evaluate damage classifier
python scripts/evaluate.py \
  --model damage \
  --weights outputs/training/damage_classifier_best.h5 \
  --data-dir data/damage_classification/test
```

---

## Running the API

### Development Mode

```bash
# Run with auto-reload
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# With custom settings
ENVIRONMENT=development \
USE_GPU=true \
LOG_LEVEL=DEBUG \
uvicorn src.api.main:app --reload
```

Access:
- API: http://localhost:8000
- Swagger Docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Production Mode

```bash
# Using gunicorn with uvicorn workers
gunicorn src.api.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120 \
  --access-logfile logs/access.log \
  --error-logfile logs/error.log
```

### Docker Deployment

```bash
# Build image
docker build -t accident-fir-api .

# Run container
docker run -d \
  --name accident-fir \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  --env-file .env \
  accident-fir-api

# View logs
docker logs -f accident-fir
```

### Docker Compose (Full Stack)

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Restart specific service
docker-compose restart api
```

Services started:
- API: http://localhost:8000
- PostgreSQL: localhost:5432
- Redis: localhost:6379
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

---

## Testing

### Unit Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html

# Run specific test file
pytest tests/test_models.py -v

# Run specific test
pytest tests/test_models.py::TestYOLODetector::test_detect_returns_list -v
```

### Integration Tests

```bash
pytest tests/integration/ -v
```

### API Tests

```bash
# Start API
uvicorn src.api.main:app &

# Run API tests
pytest tests/test_api.py -v

# Or use curl
curl -X POST "http://localhost:8000/api/v1/analyze" \
  -F "file=@test_data/accident.jpg"
```

### Load Testing

```bash
# Install locust
pip install locust

# Run load test
locust -f tests/load_test.py --host http://localhost:8000

# Access UI at http://localhost:8089
```

---

## Deployment

### AWS EC2 Deployment

```bash
# 1. Launch EC2 instance (Ubuntu 22.04, t3.large or larger)

# 2. SSH into instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# 3. Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 4. Clone repository
git clone https://github.com/yourusername/accident-fir-automation.git
cd accident-fir-automation

# 5. Configure environment
cp .env.example .env
nano .env

# 6. Deploy with Docker Compose
docker-compose up -d

# 7. Set up Nginx reverse proxy
sudo apt install nginx
sudo cp docker/nginx.conf /etc/nginx/sites-available/accident-fir
sudo ln -s /etc/nginx/sites-available/accident-fir /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### Kubernetes Deployment

```bash
# Apply configurations
kubectl apply -f k8s/

# Check status
kubectl get pods
kubectl get services

# Scale deployment
kubectl scale deployment accident-fir-api --replicas=5
```

---

## Monitoring

### Prometheus Metrics

Access Prometheus at http://localhost:9090

**Key metrics:**
- `http_requests_total`: Total HTTP requests
- `http_request_duration_seconds`: Request latency
- `model_inference_duration_seconds`: Model inference time
- `active_connections`: Active connections

### Grafana Dashboards

1. Access Grafana at http://localhost:3000
2. Login (admin/admin)
3. Import dashboard from `docker/grafana/dashboards/`

**Key dashboards:**
- API Performance
- Model Metrics
- System Resources
- Error Rates

### Application Logs

```bash
# View real-time logs
tail -f logs/app.log

# View error logs
tail -f logs/error.log

# Search logs
grep "ERROR" logs/app.log
```

---

## Troubleshooting

### Common Issues

#### 1. Model loading fails

```bash
# Check model file exists
ls -lh models/

# Download models again
python scripts/download_models.py --all

# Check file permissions
chmod 644 models/*.pt models/*.h5
```

#### 2. CUDA out of memory

```bash
# Reduce batch size in .env
BATCH_SIZE=8

# Or disable GPU
USE_GPU=false
```

#### 3. OCR not working

```bash
# Install system dependencies
sudo apt-get install tesseract-ocr
sudo apt-get install libgl1-mesa-glx

# Reinstall PaddleOCR
pip uninstall paddleocr
pip install paddleocr --no-cache-dir
```

#### 4. API timeout

```bash
# Increase timeout in uvicorn
uvicorn src.api.main:app --timeout-keep-alive 120
```

#### 5. Database connection error

```bash
# Check PostgreSQL status
docker-compose ps db

# Restart database
docker-compose restart db

# Check connection string in .env
```

### Performance Optimization

#### 1. Enable model quantization

```python
# In src/models/yolo_detector.py
model = YOLO('yolov8n.pt')
model.export(format='onnx', int8=True)
```

#### 2. Enable caching

```bash
# In .env
REDIS_URL=redis://localhost:6379/0
CACHE_TTL=3600
```

#### 3. Optimize image preprocessing

```python
# Resize large images before processing
max_size = 1280
if max(image.shape) > max_size:
    scale = max_size / max(image.shape)
    image = cv2.resize(image, None, fx=scale, fy=scale)
```

---

## Support

- **Documentation**: [docs/](docs/)
- **API Reference**: [docs/API.md](docs/API.md)
- **Issues**: https://github.com/Saivishnu4/accident-fir-automation/issues
- **Email**: saivishnusarode@gmail.com

---

## License

MIT License - see [LICENSE](LICENSE) file
