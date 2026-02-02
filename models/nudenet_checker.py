"""NudeNet-based nudity detection for post-generation safety checking."""

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image
from nudenet import NudeDetector

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class NudeNetResult:
    """Result of NudeNet analysis."""
    is_nsfw: bool
    detections: list[dict] = field(default_factory=list)
    max_score: float = 0.0
    triggered_classes: list[str] = field(default_factory=list)


class NudeNetChecker:
    """NudeNet v3 for nudity detection in generated images.
    
    Uses CNN-based detection to identify exposed body parts
    with bounding boxes and confidence scores.
    """
    
    def __init__(self):
        self.detector = None
        self._loaded = False
        self.unsafe_classes = settings.thresholds.nudenet_unsafe_classes
        self.threshold = settings.thresholds.nudenet_threshold
    
    def load(self) -> None:
        """Load NudeNet detector."""
        if self._loaded:
            return
        
        self.detector = NudeDetector()
        self._loaded = True
    
    def unload(self) -> None:
        """Unload detector."""
        if self.detector is not None:
            del self.detector
            self.detector = None
        self._loaded = False
    
    def check(
        self,
        image: Image.Image,
        threshold: float | None = None,
    ) -> NudeNetResult:
        """Check image for nudity."""
        if not self._loaded:
            self.load()
        
        threshold = threshold or self.threshold
        
        # NudeNet requires file path, save temporarily
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image.save(tmp.name)
            tmp_path = Path(tmp.name)
        
        try:
            detections = self.detector.detect(str(tmp_path))
        finally:
            tmp_path.unlink()  # Clean up temp file
        
        # Filter detections above threshold
        filtered = [
            d for d in detections 
            if d["score"] >= threshold
        ]
        
        # Check for unsafe classes
        triggered = []
        max_score = 0.0
        
        for detection in filtered:
            class_name = detection["class"]
            score = detection["score"]
            
            if score > max_score:
                max_score = score
            
            if class_name in self.unsafe_classes:
                triggered.append(class_name)
        
        is_nsfw = len(triggered) > 0
        
        return NudeNetResult(
            is_nsfw=is_nsfw,
            detections=filtered,
            max_score=max_score,
            triggered_classes=list(set(triggered)),
        )
    
    def check_batch(
        self,
        images: list[Image.Image],
        threshold: float | None = None,
    ) -> list[NudeNetResult]:
        """Check multiple images."""
        return [self.check(img, threshold) for img in images]
