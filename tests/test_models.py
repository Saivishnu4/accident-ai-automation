"""
Unit tests for Accident FIR Automation System
"""
import pytest
import numpy as np
import cv2
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.yolo_detector import YOLODetector
from src.models.ocr_engine import OCREngine
from src.models.damage_classifier import DamageClassifier


class TestYOLODetector:
    """Test YOLO detector functionality"""
    
    @pytest.fixture
    def detector(self):
        """Create detector instance"""
        # Use a dummy model path for testing
        return YOLODetector(
            model_path="yolov8n.pt",
            confidence_threshold=0.5
        )
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample test image"""
        # Create a simple BGR image
        image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        return image
    
    def test_detector_initialization(self, detector):
        """Test detector initializes correctly"""
        assert detector is not None
        assert detector.confidence_threshold == 0.5
        assert detector.model is not None
    
    def test_detect_returns_list(self, detector, sample_image):
        """Test detect method returns a list"""
        results = detector.detect(sample_image)
        assert isinstance(results, list)
    
    def test_detect_vehicle_filter(self, detector, sample_image):
        """Test vehicle detection filtering"""
        vehicles = detector.detect_vehicles(sample_image)
        assert isinstance(vehicles, list)
        # All results should be vehicles
        for v in vehicles:
            assert v["type"] == "vehicle"
    
    def test_detect_damage_filter(self, detector, sample_image):
        """Test damage detection filtering"""
        damages = detector.detect_damage(sample_image)
        assert isinstance(damages, list)
        # All results should be damage indicators
        for d in damages:
            assert d["type"] == "damage"
    
    def test_annotate_image(self, detector, sample_image):
        """Test image annotation"""
        detections = [
            {
                "type": "vehicle",
                "class": "car",
                "confidence": 0.9,
                "bbox": [100, 100, 300, 300]
            }
        ]
        annotated = detector.annotate_image(sample_image, detections)
        assert annotated.shape == sample_image.shape
        assert not np.array_equal(annotated, sample_image)  # Should be different
    
    def test_damage_severity_calculation(self, detector):
        """Test damage severity calculation"""
        detections = [
            {
                "type": "damage",
                "class": "front_damage",
                "confidence": 0.85
            },
            {
                "type": "damage",
                "class": "broken_glass",
                "confidence": 0.75
            }
        ]
        severity = detector.calculate_damage_severity(detections)
        
        assert "severity" in severity
        assert "score" in severity
        assert "damage_count" in severity
        assert severity["damage_count"] == 2
        assert severity["severity"] in ["none", "minor", "moderate", "severe", "critical"]


class TestOCREngine:
    """Test OCR engine functionality"""
    
    @pytest.fixture
    def ocr_engine(self):
        """Create OCR engine instance"""
        return OCREngine(backend="paddle", use_gpu=False)
    
    @pytest.fixture
    def sample_text_image(self):
        """Create an image with text"""
        image = np.ones((100, 400, 3), dtype=np.uint8) * 255
        cv2.putText(
            image,
            "KA 01 AB 1234",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2
        )
        return image
    
    def test_ocr_initialization(self, ocr_engine):
        """Test OCR engine initializes"""
        assert ocr_engine is not None
    
    def test_extract_text_returns_list(self, ocr_engine, sample_text_image):
        """Test text extraction returns list"""
        results = ocr_engine.extract_text(sample_text_image)
        assert isinstance(results, list)
    
    def test_license_plate_validation(self, ocr_engine):
        """Test license plate validation"""
        valid_plates = [
            "KA01AB1234",
            "DL12CD5678",
            "MH02EF9012"
        ]
        invalid_plates = [
            "ABC",
            "12345",
            "TOOLONG123456"
        ]
        
        for plate in valid_plates:
            assert ocr_engine._validate_license_plate(plate)
        
        for plate in invalid_plates:
            assert not ocr_engine._validate_license_plate(plate)
    
    def test_clean_license_plate(self, ocr_engine):
        """Test license plate cleaning"""
        dirty = "KA-01-AB-1234"
        clean = ocr_engine._clean_license_plate(dirty)
        assert "-" not in clean
        assert clean.replace(" ", "").isalnum()


class TestDamageClassifier:
    """Test damage classifier functionality"""
    
    @pytest.fixture
    def classifier(self):
        """Create classifier instance"""
        return DamageClassifier()
    
    @pytest.fixture
    def sample_image(self):
        """Create sample image"""
        return np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    
    def test_classifier_initialization(self, classifier):
        """Test classifier initializes"""
        assert classifier is not None
        assert classifier.model is not None
    
    def test_classify_returns_dict(self, classifier, sample_image):
        """Test classify returns proper dict"""
        result = classifier.classify(sample_image)
        
        assert isinstance(result, dict)
        assert "severity" in result
        assert "confidence" in result
        assert "class_id" in result
        assert "probabilities" in result
    
    def test_classify_with_bbox(self, classifier, sample_image):
        """Test classification with bounding box"""
        bbox = [50, 50, 200, 200]
        result = classifier.classify(sample_image, bbox)
        
        assert isinstance(result, dict)
        assert result["severity"] in DamageClassifier.SEVERITY_CLASSES.values()
    
    def test_severity_score_range(self, classifier, sample_image):
        """Test severity score is in valid range"""
        result = classifier.classify(sample_image)
        score = classifier.get_severity_score(result)
        
        assert 0.0 <= score <= 1.0
    
    def test_aggregate_damage_assessment(self, classifier):
        """Test damage aggregation"""
        classifications = [
            {
                "severity": "moderate",
                "confidence": 0.8,
                "class_id": 2
            },
            {
                "severity": "severe",
                "confidence": 0.9,
                "class_id": 3
            }
        ]
        
        aggregated = classifier.aggregate_damage_assessment(classifications)
        
        assert "overall_severity" in aggregated
        assert "severity_score" in aggregated
        assert "affected_areas" in aggregated
        assert aggregated["affected_areas"] == 2


class TestIntegration:
    """Integration tests for complete pipeline"""
    
    @pytest.fixture
    def sample_accident_image(self):
        """Create a sample accident scene image"""
        # Create a simple test image
        image = np.random.randint(0, 255, (1280, 720, 3), dtype=np.uint8)
        return image
    
    def test_complete_analysis_pipeline(self, sample_accident_image):
        """Test complete analysis from image to results"""
        # Initialize components
        detector = YOLODetector(model_path="yolov8n.pt")
        ocr_engine = OCREngine()
        classifier = DamageClassifier()
        
        # Detect vehicles and damage
        detections = detector.detect(sample_accident_image)
        assert isinstance(detections, list)
        
        # Calculate damage severity
        severity = detector.calculate_damage_severity(detections)
        assert isinstance(severity, dict)
        
        # Extract text (simulated)
        text_results = ocr_engine.extract_text(sample_accident_image)
        assert isinstance(text_results, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
