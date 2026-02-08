"""Utility functions for preprocessing and database operations."""
import numpy as np
from PIL import Image
import os
import subprocess
import platform
from pathlib import Path
from typing import Optional
import mediapipe as mp

#Iphone image support
from pillow_heif import register_heif_opener
register_heif_opener()

class FaceDetector:
    """Face detection and cropping using MediaPipe."""
    
    def __init__(self):
        """Initialize MediaPipe face detection."""
        self.mp_face_detection = mp.solutions.face_detection
        self.detector = self.mp_face_detection.FaceDetection(
            model_selection=1,  # 1 = full-range model (better for varied distances)
            min_detection_confidence=0.5
        )
    
    def detect_and_crop(self, image: Image.Image, padding: float = 0.2) -> Optional[Image.Image]:
        """
        Detect face and crop with padding.
        
        Args:
            image: PIL Image
            padding: Percentage padding around face bbox (0.2 = 20%)
            
        Returns:
            Cropped face image, or None if no face detected
        """
        # Convert PIL to RGB numpy array
        img_np = np.array(image)
        
        # Detect faces
        results = self.detector.process(img_np)
        
        if not results.detections: # Return original image if no face detected
            print("No face detected in the image.")
            return img_np  
        
        # Get first (most confident) face
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        
        h, w = img_np.shape[:2]
        
        # Convert relative coords to absolute
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        box_w = int(bbox.width * w)
        box_h = int(bbox.height * h)
        
        pad_w = int(box_w * padding)
        pad_h = int(box_h * padding)
        
        #Ensure bbx stays inside image
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(w, x + box_w + pad_w)
        y2 = min(h, y + box_h + pad_h)
        
        # Crop
        cropped = image.crop((x1, y1, x2, y2))
        return cropped

def open_image(image_path: str) -> Image.Image:
    return Image.open(image_path).convert("RGB")


# def preprocess_image(img: Image.Image, size: int = 224) -> Image.Image:
#     """
#     Load and preprocess image for embedding.
    
#     Args:
#         img: PIL Image object
#         size: Target size for the model to embed (default 224x224)
        
#     Returns:
#         PIL Image in RGB format cropped to standard size while maintaining aspect ratio
#     """
#     img.thumbnail((size, size), Image.BILINEAR)
#     resized_img = Image.new("RGB", (size, size), (0, 0, 0))
#     resized_img.paste(img, ((size - img.width)//2, (size - img.height)//2))

#     return resized_img

def open_file(path: str) -> None:
    """Open a file with the default application (cross-platform)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    
    system = platform.system()
    
    if system == "Darwin":  # macOS
        subprocess.run(["open", str(path)], check=False)
    elif system == "Windows":  # Windows
        os.startfile(str(path))
    elif system == "Linux":  # Linux
        subprocess.run(["xdg-open", str(path)], check=False)
    else:
        raise OSError(f"Unsupported OS: {system}")

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        a: First embedding vector
        b: Second embedding vector
        
    Returns:
        Cosine similarity score (0-1)
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))