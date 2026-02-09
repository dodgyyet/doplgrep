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
    """Face detection using MediaPipe with separate detect and crop functions."""

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

    def crop_face(self, mp_image: mp.Image, bbox: list, padding: float = 0.2) -> Image.Image:
        """Crop a face from a MediaPipe image and return as PIL.Image."""

        h, w = mp_image.height, mp_image.width

        # bbox is [x1, y1, x2, y2]
        x1, y1, x2, y2 = bbox

        # Add padding
        pad_w = int((x2 - x1) * padding)
        pad_h = int((y2 - y1) * padding)
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(w, x2 + pad_w)
        y2 = min(h, y2 + pad_h)

        # Convert MediaPipe image -> NumPy array for cropping
        img_np = np.array(mp_image.numpy_view())  # HWC RGB uint8
        cropped_np = img_np[y1:y2, x1:x2, :]

        # Convert to PIL for DINOv3 embedding
        pil_img = Image.fromarray(cropped_np)
        pil_img.show()
        return pil_img.convert("RGB")

    def detect_and_crop(self, image_path: str, padding: float = 0.2) -> Optional[Image.Image]:
        """Detect and crop the first face from an image file."""
        mp_image = mp.Image.create_from_file(image_path)
        bbx = self.detect_faces(mp_image)

        if not bbx:
            print("No face detected, returning original image.")
            return open_image(image_path)

    
        return self.crop_face(mp_image, bbx, padding=padding)

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


