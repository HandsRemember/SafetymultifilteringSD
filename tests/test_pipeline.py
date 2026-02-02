"""Unit tests for the safety pipeline components."""

import pytest
from PIL import Image
import numpy as np


class TestImageUtils:
    """Tests for image utility functions."""
    
    def test_apply_blur(self):
        """Test Gaussian blur application."""
        from utils.image_utils import apply_blur
        
        # Create test image
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        
        blurred = apply_blur(img, kernel_size=15, sigma=5.0)
        
        assert blurred.size == img.size
        assert isinstance(blurred, Image.Image)
    
    def test_resize_image(self):
        """Test image resizing."""
        from utils.image_utils import resize_image
        
        img = Image.fromarray(np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8))
        
        resized = resize_image(img, width=150)
        
        assert resized.width == 150
        assert resized.height == 100  # Maintains aspect ratio


class TestMetrics:
    """Tests for benchmark metrics."""
    
    def test_timing_decorator(self):
        """Test timing decorator."""
        from utils.metrics import timing
        import time
        
        @timing
        def slow_function():
            time.sleep(0.01)
            return "done"
        
        result, elapsed = slow_function()
        
        assert result == "done"
        assert elapsed >= 10  # At least 10ms
    
    def test_safety_metrics(self):
        """Test safety metrics calculation."""
        from utils.metrics import SafetyMetrics
        
        metrics = SafetyMetrics()
        
        # Add some results
        metrics.update(predicted_unsafe=True, actual_unsafe=True)  # TP
        metrics.update(predicted_unsafe=False, actual_unsafe=False)  # TN
        metrics.update(predicted_unsafe=True, actual_unsafe=False)  # FP
        metrics.update(predicted_unsafe=False, actual_unsafe=True)  # FN
        
        assert metrics.total == 4
        assert metrics.accuracy == 0.5
        assert metrics.precision == 0.5
        assert metrics.recall == 0.5
    
    def test_benchmark_metrics(self):
        """Test benchmark metrics aggregation."""
        from utils.metrics import BenchmarkMetrics
        
        metrics = BenchmarkMetrics()
        
        metrics.record_layer_time("layer1", 100.0)
        metrics.record_layer_time("layer1", 150.0)
        metrics.record_layer_time("layer2", 50.0)
        
        metrics.record_pipeline_run(200.0)
        metrics.record_pipeline_run(250.0)
        
        summary = metrics.get_summary()
        
        assert summary["total_images"] == 2
        assert summary["layers"]["layer1"]["mean_ms"] == 125.0


class TestConfig:
    """Tests for configuration."""
    
    def test_settings_defaults(self):
        """Test default settings values."""
        from config import settings
        
        assert settings.device == "cuda"
        assert settings.models.diffusion_id == "runwayml/stable-diffusion-v1-5"
        assert settings.thresholds.nudenet_threshold == 0.6
    
    def test_path_config(self):
        """Test path configuration."""
        from config.settings import PathConfig
        
        paths = PathConfig()
        
        assert paths.output_dir.name == "outputs"
        assert paths.cache_dir.name == ".cache"


# Integration tests (require GPU and models)
@pytest.mark.skipif(True, reason="Requires GPU and downloaded models")
class TestIntegration:
    """Integration tests for full pipeline."""
    
    def test_full_pipeline(self):
        """Test full pipeline execution."""
        from pipeline import SafetyPipeline, SafetyDecision
        
        pipeline = SafetyPipeline(mode="full")
        result = pipeline.run("A cat sitting on a couch", save_output=False)
        
        assert result.decision in [SafetyDecision.SAFE, SafetyDecision.UNSAFE_BLURRED]
        assert result.image is not None
