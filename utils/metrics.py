"""Timing decorators and benchmark metrics for pipeline evaluation."""

import time
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable


def timing(func: Callable) -> Callable:
    """Decorator to measure function execution time in milliseconds."""
    @wraps(func)
    def wrapper(*args, **kwargs) -> tuple[Any, float]:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return result, elapsed_ms
    return wrapper


def timing_result_only(func: Callable) -> Callable:
    """Decorator that returns only the result, storing time internally."""
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        wrapper.last_elapsed_ms = (time.perf_counter() - start) * 1000
        return result
    wrapper.last_elapsed_ms = 0.0
    return wrapper


@dataclass
class LayerMetrics:
    """Metrics for a single pipeline layer."""
    name: str
    latencies: list[float] = field(default_factory=list)
    
    @property
    def count(self) -> int:
        return len(self.latencies)
    
    @property
    def mean_ms(self) -> float:
        if not self.latencies:
            return 0.0
        return sum(self.latencies) / len(self.latencies)
    
    @property
    def min_ms(self) -> float:
        return min(self.latencies) if self.latencies else 0.0
    
    @property
    def max_ms(self) -> float:
        return max(self.latencies) if self.latencies else 0.0
    
    def add(self, latency_ms: float) -> None:
        self.latencies.append(latency_ms)
    
    def reset(self) -> None:
        self.latencies.clear()


@dataclass
class SafetyMetrics:
    """Safety evaluation metrics."""
    true_positives: int = 0  # Correctly identified unsafe
    true_negatives: int = 0  # Correctly identified safe
    false_positives: int = 0  # Safe marked as unsafe
    false_negatives: int = 0  # Unsafe marked as safe
    
    @property
    def total(self) -> int:
        return self.true_positives + self.true_negatives + self.false_positives + self.false_negatives
    
    @property
    def accuracy(self) -> float:
        if self.total == 0:
            return 0.0
        return (self.true_positives + self.true_negatives) / self.total
    
    @property
    def precision(self) -> float:
        """Of all predicted unsafe, how many were actually unsafe."""
        predicted_positive = self.true_positives + self.false_positives
        if predicted_positive == 0:
            return 0.0
        return self.true_positives / predicted_positive
    
    @property
    def recall(self) -> float:
        """Of all actually unsafe, how many did we catch (Safety Recall)."""
        actual_positive = self.true_positives + self.false_negatives
        if actual_positive == 0:
            return 0.0
        return self.true_positives / actual_positive
    
    @property
    def f1_score(self) -> float:
        p, r = self.precision, self.recall
        if p + r == 0:
            return 0.0
        return 2 * (p * r) / (p + r)
    
    @property
    def false_positive_rate(self) -> float:
        """Of all actually safe, how many were incorrectly flagged."""
        actual_negative = self.true_negatives + self.false_positives
        if actual_negative == 0:
            return 0.0
        return self.false_positives / actual_negative
    
    def update(self, predicted_unsafe: bool, actual_unsafe: bool) -> None:
        if predicted_unsafe and actual_unsafe:
            self.true_positives += 1
        elif not predicted_unsafe and not actual_unsafe:
            self.true_negatives += 1
        elif predicted_unsafe and not actual_unsafe:
            self.false_positives += 1
        else:
            self.false_negatives += 1
    
    def reset(self) -> None:
        self.true_positives = 0
        self.true_negatives = 0
        self.false_positives = 0
        self.false_negatives = 0


class BenchmarkMetrics:
    """Aggregated benchmark metrics for pipeline evaluation."""
    
    def __init__(self):
        self.layers: dict[str, LayerMetrics] = {}
        self.safety = SafetyMetrics()
        self.total_images = 0
        self.total_time_ms = 0.0
    
    def add_layer(self, name: str) -> LayerMetrics:
        if name not in self.layers:
            self.layers[name] = LayerMetrics(name=name)
        return self.layers[name]
    
    def record_layer_time(self, name: str, latency_ms: float) -> None:
        layer = self.add_layer(name)
        layer.add(latency_ms)
    
    def record_pipeline_run(self, total_ms: float) -> None:
        self.total_images += 1
        self.total_time_ms += total_ms
    
    @property
    def throughput(self) -> float:
        """Images per second."""
        if self.total_time_ms == 0:
            return 0.0
        return self.total_images / (self.total_time_ms / 1000)
    
    @property
    def mean_latency_ms(self) -> float:
        if self.total_images == 0:
            return 0.0
        return self.total_time_ms / self.total_images
    
    def get_summary(self) -> dict:
        """Get summary of all metrics."""
        layer_summary = {
            name: {
                "mean_ms": layer.mean_ms,
                "min_ms": layer.min_ms,
                "max_ms": layer.max_ms,
                "count": layer.count,
            }
            for name, layer in self.layers.items()
        }
        
        return {
            "total_images": self.total_images,
            "total_time_ms": self.total_time_ms,
            "mean_latency_ms": self.mean_latency_ms,
            "throughput_ips": self.throughput,
            "layers": layer_summary,
            "safety": {
                "accuracy": self.safety.accuracy,
                "precision": self.safety.precision,
                "recall": self.safety.recall,
                "f1_score": self.safety.f1_score,
                "false_positive_rate": self.safety.false_positive_rate,
            },
        }
    
    def reset(self) -> None:
        for layer in self.layers.values():
            layer.reset()
        self.safety.reset()
        self.total_images = 0
        self.total_time_ms = 0.0
