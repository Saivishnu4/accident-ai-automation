"""
YOLO-based Vehicle and Damage Detection Module
"""
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Tuple, Optional
import torch
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class YOLODetector:
    """
    YOLOv8-based detector for vehicles and accident-related objects
    """
    
    def __init__(
        self,
        model_path: str = "models/yolov8_accident.pt",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize YOLO detector
        
        Args:
            model_path: Path to trained YOLO model
            confidence_threshold: Minimum confidence score for detections
            iou_threshold: IoU threshold for NMS
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        
        # Load model
        try:
            self.model = YOLO(str(self.model_path))
            self.model.to(self.device)
            logger.info(f"YOLO model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
        
        # Vehicle classes (customize based on your training)
        self.vehicle_classes = {
            0: "car",
            1: "motorcycle",
            2: "truck",
            3: "bus",
            4: "bicycle"
        }
        
        # Damage indicators
        self.damage_classes = {
            5: "front_damage",
            6: "rear_damage",
            7: "side_damage",
            8: "broken_glass",
            9: "deployed_airbag"
        }
    
    def detect(
        self,
        image: np.ndarray,
        conf_threshold: Optional[float] = None
    ) -> List[Dict]:
        """
        Detect vehicles and damage in image
        
        Args:
            image: Input image (BGR format)
            conf_threshold: Override default confidence threshold
            
        Returns:
            List of detection dictionaries
        """
        if conf_threshold is None:
            conf_threshold = self.confidence_threshold
        
        try:
            # Run inference
            results = self.model.predict(
                image,
                conf=conf_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False
            )
            
            detections = []
            
            for result in results:
                boxes = result.boxes
                
                for i, box in enumerate(boxes):
                    # Extract box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    
                    # Determine object type
                    if class_id in self.vehicle_classes:
                        obj_type = "vehicle"
                        obj_class = self.vehicle_classes[class_id]
                    elif class_id in self.damage_classes:
                        obj_type = "damage"
                        obj_class = self.damage_classes[class_id]
                    else:
                        obj_type = "unknown"
                        obj_class = f"class_{class_id}"
                    
                    detection = {
                        "type": obj_type,
                        "class": obj_class,
                        "confidence": confidence,
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "center": [(int(x1) + int(x2)) // 2, (int(y1) + int(y2)) // 2]
                    }
                    
                    detections.append(detection)
            
            logger.info(f"Detected {len(detections)} objects")
            return detections
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            raise
    
    def detect_vehicles(self, image: np.ndarray) -> List[Dict]:
        """
        Detect only vehicles in the image
        """
        all_detections = self.detect(image)
        return [d for d in all_detections if d["type"] == "vehicle"]
    
    def detect_damage(self, image: np.ndarray) -> List[Dict]:
        """
        Detect only damage indicators in the image
        """
        all_detections = self.detect(image)
        return [d for d in all_detections if d["type"] == "damage"]
    
    def annotate_image(
        self,
        image: np.ndarray,
        detections: List[Dict],
        show_labels: bool = True
    ) -> np.ndarray:
        """
        Draw bounding boxes and labels on image
        
        Args:
            image: Input image
            detections: List of detections
            show_labels: Whether to show class labels
            
        Returns:
            Annotated image
        """
        annotated = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            confidence = det["confidence"]
            label = det["class"]
            
            # Color coding
            if det["type"] == "vehicle":
                color = (0, 255, 0)  # Green
            elif det["type"] == "damage":
                color = (0, 0, 255)  # Red
            else:
                color = (255, 0, 0)  # Blue
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            if show_labels:
                # Draw label
                label_text = f"{label}: {confidence:.2f}"
                (text_width, text_height), baseline = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                
                cv2.rectangle(
                    annotated,
                    (x1, y1 - text_height - baseline - 5),
                    (x1 + text_width, y1),
                    color,
                    -1
                )
                
                cv2.putText(
                    annotated,
                    label_text,
                    (x1, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )
        
        return annotated
    
    def calculate_damage_severity(self, detections: List[Dict]) -> Dict:
        """
        Calculate overall damage severity based on detections
        
        Returns:
            Dictionary with severity assessment
        """
        damage_detections = [d for d in detections if d["type"] == "damage"]
        
        if not damage_detections:
            return {
                "severity": "none",
                "score": 0.0,
                "damage_count": 0
            }
        
        # Calculate severity score
        severity_weights = {
            "front_damage": 0.3,
            "rear_damage": 0.3,
            "side_damage": 0.25,
            "broken_glass": 0.15,
            "deployed_airbag": 0.4  # Indicates significant impact
        }
        
        total_score = 0.0
        for det in damage_detections:
            weight = severity_weights.get(det["class"], 0.2)
            total_score += det["confidence"] * weight
        
        # Normalize and categorize
        normalized_score = min(total_score / len(damage_detections), 1.0)
        
        if normalized_score < 0.2:
            severity = "minor"
        elif normalized_score < 0.5:
            severity = "moderate"
        elif normalized_score < 0.8:
            severity = "severe"
        else:
            severity = "critical"
        
        return {
            "severity": severity,
            "score": float(normalized_score),
            "damage_count": len(damage_detections),
            "damage_types": list(set(d["class"] for d in damage_detections))
        }
