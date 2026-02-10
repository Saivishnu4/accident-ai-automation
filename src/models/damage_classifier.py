"""
Damage Severity Classification Module
"""
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Dict, List, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DamageClassifier:
    """
    CNN-based classifier for assessing vehicle damage severity
    """
    
    # Damage severity classes
    SEVERITY_CLASSES = {
        0: "none",
        1: "minor",
        2: "moderate",
        3: "severe",
        4: "total"
    }
    
    def __init__(
        self,
        model_path: str = "models/damage_classifier.h5",
        input_size: Tuple[int, int] = (224, 224)
    ):
        """
        Initialize damage classifier
        
        Args:
            model_path: Path to trained Keras model
            input_size: Input image size (height, width)
        """
        self.model_path = Path(model_path)
        self.input_size = input_size
        
        # Load model
        try:
            self.model = keras.models.load_model(str(self.model_path))
            logger.info(f"Damage classifier loaded from {self.model_path}")
        except Exception as e:
            logger.warning(f"Could not load model: {e}. Using dummy classifier.")
            self.model = self._build_dummy_model()
    
    def _build_dummy_model(self):
        """Build a simple model architecture for testing"""
        model = keras.Sequential([
            keras.layers.Input(shape=(*self.input_size, 3)),
            keras.layers.Conv2D(32, 3, activation='relu'),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(64, 3, activation='relu'),
            keras.layers.MaxPooling2D(),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(len(self.SEVERITY_CLASSES), activation='softmax')
        ])
        return model
    
    def classify(
        self,
        image: np.ndarray,
        bbox: List[int] = None
    ) -> Dict:
        """
        Classify damage severity in image
        
        Args:
            image: Input image (BGR format)
            bbox: Optional bounding box [x1, y1, x2, y2] to focus on
            
        Returns:
            Classification results with confidence scores
        """
        # Crop to bbox if provided
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            image = image[y1:y2, x1:x2]
        
        # Preprocess image
        processed = self._preprocess(image)
        
        # Make prediction
        predictions = self.model.predict(processed, verbose=0)[0]
        
        # Get top class and confidence
        predicted_class = int(np.argmax(predictions))
        confidence = float(predictions[predicted_class])
        severity = self.SEVERITY_CLASSES[predicted_class]
        
        # Get all class probabilities
        class_probabilities = {
            self.SEVERITY_CLASSES[i]: float(predictions[i])
            for i in range(len(self.SEVERITY_CLASSES))
        }
        
        result = {
            "severity": severity,
            "confidence": confidence,
            "class_id": predicted_class,
            "probabilities": class_probabilities
        }
        
        logger.info(f"Classified damage as '{severity}' with {confidence:.2%} confidence")
        return result
    
    def classify_multiple(
        self,
        image: np.ndarray,
        bboxes: List[List[int]]
    ) -> List[Dict]:
        """
        Classify damage severity for multiple regions
        
        Args:
            image: Input image
            bboxes: List of bounding boxes
            
        Returns:
            List of classification results
        """
        results = []
        for bbox in bboxes:
            result = self.classify(image, bbox)
            result["bbox"] = bbox
            results.append(result)
        return results
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for classification
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Preprocessed image ready for model
        """
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        resized = cv2.resize(image, self.input_size)
        
        # Normalize pixel values
        normalized = resized.astype(np.float32) / 255.0
        
        # Add batch dimension
        batched = np.expand_dims(normalized, axis=0)
        
        return batched
    
    def get_severity_score(self, classification: Dict) -> float:
        """
        Convert classification to numerical severity score (0-1)
        
        Args:
            classification: Classification result dictionary
            
        Returns:
            Severity score between 0 (no damage) and 1 (total loss)
        """
        severity_values = {
            "none": 0.0,
            "minor": 0.25,
            "moderate": 0.5,
            "severe": 0.75,
            "total": 1.0
        }
        
        severity = classification["severity"]
        base_score = severity_values.get(severity, 0.5)
        
        # Adjust based on confidence
        confidence = classification["confidence"]
        adjusted_score = base_score * (0.5 + 0.5 * confidence)
        
        return adjusted_score
    
    def aggregate_damage_assessment(
        self,
        classifications: List[Dict]
    ) -> Dict:
        """
        Aggregate multiple damage classifications into overall assessment
        
        Args:
            classifications: List of classification results
            
        Returns:
            Aggregated damage assessment
        """
        if not classifications:
            return {
                "overall_severity": "none",
                "severity_score": 0.0,
                "affected_areas": 0,
                "breakdown": {}
            }
        
        # Calculate average severity score
        scores = [self.get_severity_score(c) for c in classifications]
        avg_score = np.mean(scores)
        max_score = np.max(scores)
        
        # Count severity levels
        severity_counts = {}
        for c in classifications:
            sev = c["severity"]
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
        
        # Determine overall severity (use max severity found)
        severity_order = ["none", "minor", "moderate", "severe", "total"]
        max_severity = "none"
        for sev in reversed(severity_order):
            if severity_counts.get(sev, 0) > 0:
                max_severity = sev
                break
        
        # If multiple severe damages, escalate assessment
        if severity_counts.get("severe", 0) >= 2:
            max_severity = "total"
        
        return {
            "overall_severity": max_severity,
            "severity_score": float(max_score),
            "average_score": float(avg_score),
            "affected_areas": len(classifications),
            "breakdown": severity_counts,
            "confidence": float(np.mean([c["confidence"] for c in classifications]))
        }


def build_damage_classifier_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 5
) -> keras.Model:
    """
    Build ResNet50-based damage classification model
    
    Args:
        input_shape: Input image shape
        num_classes: Number of severity classes
        
    Returns:
        Compiled Keras model
    """
    # Load pre-trained ResNet50
    base_model = keras.applications.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        pooling='avg'
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Build classifier head
    inputs = keras.Input(shape=input_shape)
    x = keras.applications.resnet50.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=2)]
    )
    
    logger.info("Damage classifier model built successfully")
    return model
