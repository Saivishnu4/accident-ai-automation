"""
OCR Engine for License Plate and Text Extraction
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import re
import logging
from paddleocr import PaddleOCR
import easyocr

logger = logging.getLogger(__name__)


class OCREngine:
    """
    Multi-backend OCR engine for text extraction from accident images
    Primarily focused on license plate recognition
    """
    
    def __init__(
        self,
        backend: str = "paddle",
        use_gpu: bool = False,
        lang: List[str] = ["en"]
    ):
        """
        Initialize OCR engine
        
        Args:
            backend: OCR backend to use ('paddle', 'easy', or 'both')
            use_gpu: Whether to use GPU acceleration
            lang: Languages to recognize
        """
        self.backend = backend
        self.use_gpu = use_gpu
        self.lang = lang
        
        # Initialize OCR engines
        if backend in ["paddle", "both"]:
            try:
                self.paddle_ocr = PaddleOCR(
                    use_angle_cls=True,
                    lang='en',
                    use_gpu=use_gpu,
                    show_log=False
                )
                logger.info("PaddleOCR initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize PaddleOCR: {e}")
                self.paddle_ocr = None
        
        if backend in ["easy", "both"]:
            try:
                self.easy_reader = easyocr.Reader(
                    lang_list=lang,
                    gpu=use_gpu,
                    verbose=False
                )
                logger.info("EasyOCR initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize EasyOCR: {e}")
                self.easy_reader = None
        
        # License plate patterns for validation
        self.license_patterns = [
            r'[A-Z]{2}\s*\d{2}\s*[A-Z]{1,2}\s*\d{4}',  # Indian: KA 01 AB 1234
            r'[A-Z]{2}\s*\d{1,2}\s*[A-Z]{3}\s*\d{4}',  # Alternative Indian
            r'[A-Z]{3}\s*\d{3,4}',  # Simple format
            r'\d[A-Z]{3}\d{3}',  # US style
        ]
    
    def extract_text(
        self,
        image: np.ndarray,
        preprocessing: bool = True
    ) -> List[Dict]:
        """
        Extract all text from image
        
        Args:
            image: Input image (BGR format)
            preprocessing: Apply preprocessing for better OCR
            
        Returns:
            List of detected text regions with confidence
        """
        if preprocessing:
            image = self._preprocess_image(image)
        
        results = []
        
        # Use PaddleOCR
        if self.paddle_ocr is not None:
            try:
                paddle_results = self.paddle_ocr.ocr(image, cls=True)
                if paddle_results and paddle_results[0]:
                    for line in paddle_results[0]:
                        bbox = line[0]
                        text, conf = line[1]
                        results.append({
                            "text": text,
                            "confidence": conf,
                            "bbox": self._normalize_bbox(bbox),
                            "source": "paddle"
                        })
            except Exception as e:
                logger.error(f"PaddleOCR extraction failed: {e}")
        
        # Use EasyOCR
        if self.easy_reader is not None:
            try:
                easy_results = self.easy_reader.readtext(image)
                for bbox, text, conf in easy_results:
                    results.append({
                        "text": text,
                        "confidence": conf,
                        "bbox": self._normalize_bbox(bbox),
                        "source": "easy"
                    })
            except Exception as e:
                logger.error(f"EasyOCR extraction failed: {e}")
        
        # Deduplicate and merge results
        results = self._merge_duplicate_detections(results)
        
        logger.info(f"Extracted {len(results)} text regions")
        return results
    
    def extract_license_plates(
        self,
        image: np.ndarray,
        vehicle_bboxes: Optional[List[List[int]]] = None
    ) -> List[Dict]:
        """
        Extract license plate numbers from image
        
        Args:
            image: Input image
            vehicle_bboxes: Optional vehicle bounding boxes to focus search
            
        Returns:
            List of detected license plates
        """
        # If vehicle bboxes provided, crop and process each
        if vehicle_bboxes:
            license_plates = []
            for bbox in vehicle_bboxes:
                x1, y1, x2, y2 = bbox
                # Expand bbox slightly to capture license plate
                h, w = image.shape[:2]
                x1 = max(0, x1 - 20)
                y1 = max(0, y1 - 20)
                x2 = min(w, x2 + 20)
                y2 = min(h, y2 + 20)
                
                vehicle_crop = image[y1:y2, x1:x2]
                plates = self._extract_plates_from_crop(vehicle_crop, (x1, y1))
                license_plates.extend(plates)
        else:
            # Process entire image
            license_plates = self._extract_plates_from_crop(image)
        
        # Validate and clean license plates
        validated_plates = []
        for plate in license_plates:
            cleaned = self._clean_license_plate(plate["text"])
            if self._validate_license_plate(cleaned):
                plate["text"] = cleaned
                validated_plates.append(plate)
        
        logger.info(f"Found {len(validated_plates)} valid license plates")
        return validated_plates
    
    def _extract_plates_from_crop(
        self,
        crop: np.ndarray,
        offset: Tuple[int, int] = (0, 0)
    ) -> List[Dict]:
        """Extract potential license plates from image crop"""
        # Enhanced preprocessing for license plates
        processed = self._preprocess_license_plate(crop)
        
        # Extract text
        text_results = self.extract_text(processed, preprocessing=False)
        
        plates = []
        for result in text_results:
            # Adjust bbox coordinates based on offset
            bbox = result["bbox"]
            adjusted_bbox = [
                bbox[0] + offset[0],
                bbox[1] + offset[1],
                bbox[2] + offset[0],
                bbox[3] + offset[1]
            ]
            
            plates.append({
                "text": result["text"],
                "confidence": result["confidence"],
                "bbox": adjusted_bbox
            })
        
        return plates
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """General image preprocessing for OCR"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )
        
        return thresh
    
    def _preprocess_license_plate(self, image: np.ndarray) -> np.ndarray:
        """Specialized preprocessing for license plate images"""
        # Resize for better OCR
        height, width = image.shape[:2]
        if height < 100:
            scale = 100 / height
            image = cv2.resize(image, None, fx=scale, fy=scale)
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Bilateral filter to reduce noise while keeping edges
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel)
        
        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            morph,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )
        
        return thresh
    
    def _clean_license_plate(self, text: str) -> str:
        """Clean and standardize license plate text"""
        # Remove common OCR errors
        text = text.upper()
        text = re.sub(r'[^A-Z0-9\s]', '', text)
        
        # Common character substitutions
        substitutions = {
            'O': '0',
            'I': '1',
            'S': '5',
            'B': '8'
        }
        
        # Apply substitutions in digit positions
        cleaned = ""
        for char in text:
            if char.isdigit() or char.isalpha():
                cleaned += char
            elif char == ' ':
                cleaned += ' '
        
        return cleaned.strip()
    
    def _validate_license_plate(self, text: str) -> bool:
        """Validate if text matches license plate patterns"""
        text = re.sub(r'\s+', '', text)  # Remove spaces for pattern matching
        
        for pattern in self.license_patterns:
            if re.match(pattern, text):
                return True
        
        # Basic validation: should have both letters and numbers
        has_letters = any(c.isalpha() for c in text)
        has_digits = any(c.isdigit() for c in text)
        proper_length = 6 <= len(text) <= 12
        
        return has_letters and has_digits and proper_length
    
    def _normalize_bbox(self, bbox) -> List[int]:
        """Convert various bbox formats to [x1, y1, x2, y2]"""
        if isinstance(bbox, list):
            if len(bbox) == 4 and all(isinstance(x, (int, float)) for x in bbox):
                return [int(x) for x in bbox]
            elif len(bbox) == 4 and isinstance(bbox[0], (list, tuple)):
                # Convert from [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] format
                x_coords = [p[0] for p in bbox]
                y_coords = [p[1] for p in bbox]
                return [
                    int(min(x_coords)),
                    int(min(y_coords)),
                    int(max(x_coords)),
                    int(max(y_coords))
                ]
        return [0, 0, 0, 0]
    
    def _merge_duplicate_detections(
        self,
        results: List[Dict],
        iou_threshold: float = 0.5
    ) -> List[Dict]:
        """Merge duplicate detections from different OCR backends"""
        if len(results) <= 1:
            return results
        
        merged = []
        used = set()
        
        for i, r1 in enumerate(results):
            if i in used:
                continue
            
            duplicates = [r1]
            for j, r2 in enumerate(results[i+1:], start=i+1):
                if j in used:
                    continue
                
                # Calculate IoU
                iou = self._calculate_iou(r1["bbox"], r2["bbox"])
                if iou > iou_threshold:
                    duplicates.append(r2)
                    used.add(j)
            
            # Merge duplicates - keep highest confidence
            best = max(duplicates, key=lambda x: x["confidence"])
            merged.append(best)
        
        return merged
    
    def _calculate_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """Calculate Intersection over Union between two bboxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
