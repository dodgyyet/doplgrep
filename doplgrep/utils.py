"""Utility functions for preprocessing and database operations."""
import numpy as np
from PIL import Image
import os
import subprocess
import platform
from pathlib import Path
from typing import Optional
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

#Supress alerts 
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

#Iphone image support
from pillow_heif import register_heif_opener
register_heif_opener()


class FaceDetector:
    """Face detection using MediaPipe with EXIF-aware cropping."""

    def __init__(self):
        """Initialize MediaPipe face detector."""
        model_path = Path(__file__).parent / "blaze_face_short_range.tflite"
        BaseOptions = mp.tasks.BaseOptions
        FaceDetector = mp.tasks.vision.FaceDetector
        FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=VisionRunningMode.IMAGE
        )

        self.detector = FaceDetector.create_from_options(options)

    def detect_faces(self, mp_image: mp.Image) -> list:
        """Return list of bounding boxes for faces detected in a MediaPipe image."""
        result = self.detector.detect(mp_image)
        if not result.detections:
            return []
        detection = result.detections[0]      
        if not hasattr(detection, "bounding_box") or detection.bounding_box is None:
            return [] 
        bbx = detection.bounding_box  
        x1 = int(bbx.origin_x)
        y1 = int(bbx.origin_y)
        x2 = x1 + int(bbx.width)
        y2 = y1 + int(bbx.height)
        return [x1, y1, x2, y2]

    def crop_face(self, pil_image: Image.Image, bbox: list, 
                  padding_top: float = 0.8, 
                  padding_sides: float = 0.4, 
                  padding_bottom: float = 0.1) -> Image.Image:
        """
        Crop a face from a PIL Image with asymmetric padding.
        
        Args:
            pil_image: PIL Image (already rotated correctly via EXIF)
            bbox: Bounding box [x1, y1, x2, y2]
            padding_top: Padding multiplier for top (default 0.8 = 80% extra for hair)
            padding_sides: Padding multiplier for left/right (default 0.4)
            padding_bottom: Padding multiplier for bottom (default 0.3)
            
        Returns:
            PIL Image in RGB format
        """
        w, h = pil_image.size  # PIL uses (width, height)

        # bbox is [x1, y1, x2, y2]
        x1, y1, x2, y2 = bbox
        
        face_width = x2 - x1
        face_height = y2 - y1

        # Asymmetric padding - MUCH more on top for hair and forehead
        pad_w = int(face_width * padding_sides)
        pad_h_top = int(face_height * padding_top)
        pad_h_bottom = int(face_height * padding_bottom)
        
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h_top)
        x2 = min(w, x2 + pad_w)
        y2 = min(h, y2 + pad_h_bottom)

        # Crop the PIL image directly (no numpy conversion needed)
        cropped = pil_image.crop((x1, y1, x2, y2))
        
        # Ensure RGB format
        return cropped.convert("RGB")

    def detect_and_crop(self, image_path: str, 
                       padding_top: float = 0.8,
                       padding_sides: float = 0.4,
                       padding_bottom: float = 0.3,
                       fallback_to_full: bool = True) -> Optional[Image.Image]:
        """
        Detect and crop the first face from an image file.
        FIXED: Now handles EXIF rotation properly to prevent 270° rotation.
        
        Args:
            image_path: Path to image file
            padding_top: Extra space above face (0.8 = 80% of face height for hair)
            padding_sides: Extra space on left/right (0.4 = 40%)
            padding_bottom: Extra space below face (0.3 = 30%)
            fallback_to_full: If no face detected, return full image (vs None)
            
        Returns:
            Cropped face as PIL Image, or full image if no face detected
        """
        from PIL import ImageOps
        
        # Load image with PIL first and handle EXIF rotation
        pil_image = Image.open(image_path)
        
        # CRITICAL FIX: Handle EXIF orientation (fixes 270° rotation)
        pil_image = ImageOps.exif_transpose(pil_image)
        if pil_image is None:  # If no EXIF, exif_transpose returns None
            pil_image = Image.open(image_path)
        
        # Convert to RGB to ensure 3 channels
        pil_image = pil_image.convert("RGB")
        
        # Now convert to MediaPipe format for face detection
        # We'll use the PIL image directly for cropping to preserve orientation
        img_np = np.array(pil_image)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_np)
        
        # Detect face
        bbx = self.detect_faces(mp_image)

        if not bbx:
            if fallback_to_full:
                print(f"No face detected in {image_path}, using full image")
                return pil_image
            else:
                return None

        # Crop using PIL image (not MediaPipe) to preserve correct orientation
        return self.crop_face(pil_image, bbx, 
                            padding_top=padding_top,
                            padding_sides=padding_sides, 
                            padding_bottom=padding_bottom)
    
def open_image(image_path: str) -> Image.Image:
    return Image.open(image_path).convert("RGB")


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

#test script
if __name__ == "__main__":
    # Example usage
    img_path = "../input_images/dorian.heic"
    
    detector = FaceDetector()
    cropped_img = detector.detect_and_crop(img_path)
    
    if cropped_img is not None:
        cropped_img.show()


