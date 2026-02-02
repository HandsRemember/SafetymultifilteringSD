"""Image utility functions for blur, save, and load operations."""

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from config import settings


def apply_blur(
    image: Image.Image,
    kernel_size: int | None = None,
    sigma: float | None = None,
) -> Image.Image:
    """Apply Gaussian blur to an image."""
    kernel_size = kernel_size or settings.blur_kernel_size
    sigma = sigma or settings.blur_sigma
    
    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Convert PIL to numpy
    img_array = np.array(image)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(img_array, (kernel_size, kernel_size), sigma)
    
    return Image.fromarray(blurred)


def apply_region_blur(
    image: Image.Image,
    regions: list[tuple[int, int, int, int]],
    kernel_size: int | None = None,
    sigma: float | None = None,
) -> Image.Image:
    """Apply blur only to specific regions (bounding boxes)."""
    kernel_size = kernel_size or settings.blur_kernel_size
    sigma = sigma or settings.blur_sigma
    
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    img_array = np.array(image)
    
    for x1, y1, x2, y2 in regions:
        # Ensure valid bounds
        h, w = img_array.shape[:2]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        
        # Extract region
        region = img_array[y1:y2, x1:x2]
        
        # Blur region
        blurred_region = cv2.GaussianBlur(region, (kernel_size, kernel_size), sigma)
        
        # Replace in image
        img_array[y1:y2, x1:x2] = blurred_region
    
    return Image.fromarray(img_array)


def save_image(
    image: Image.Image,
    filename: str | None = None,
    output_dir: Path | None = None,
    prefix: str = "",
    suffix: str = "",
    subfolder: str | None = None,
) -> Path:
    """Save image to output directory."""
    output_dir = output_dir or settings.paths.output_dir
    
    # Support subfolders (e.g., 'original', 'blurred')
    if subfolder:
        output_dir = output_dir / subfolder
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}{timestamp}{suffix}.png"
    
    output_path = output_dir / filename
    image.save(output_path)
    
    return output_path


def save_image_pair(
    original: Image.Image,
    blurred: Image.Image,
    base_name: str | None = None,
    output_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Save both original and blurred images with same name in different folders."""
    output_dir = output_dir or settings.paths.output_dir
    
    if base_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{timestamp}.png"
    
    original_path = save_image(original, filename=base_name, output_dir=output_dir, subfolder="original")
    blurred_path = save_image(blurred, filename=base_name, output_dir=output_dir, subfolder="blurred")
    
    return original_path, blurred_path


def load_image(path: str | Path) -> Image.Image:
    """Load image from path."""
    return Image.open(path).convert("RGB")


def resize_image(
    image: Image.Image,
    width: int | None = None,
    height: int | None = None,
    maintain_aspect: bool = True,
) -> Image.Image:
    """Resize image."""
    if width is None and height is None:
        return image
    
    original_width, original_height = image.size
    
    if maintain_aspect:
        if width and height:
            # Fit within bounds
            ratio = min(width / original_width, height / original_height)
            new_width = int(original_width * ratio)
            new_height = int(original_height * ratio)
        elif width:
            ratio = width / original_width
            new_width = width
            new_height = int(original_height * ratio)
        else:
            ratio = height / original_height
            new_width = int(original_width * ratio)
            new_height = height
    else:
        new_width = width or original_width
        new_height = height or original_height
    
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def create_comparison_grid(
    images: list[Image.Image],
    labels: list[str] | None = None,
    cols: int = 2,
) -> Image.Image:
    """Create a grid of images for comparison."""
    if not images:
        raise ValueError("No images provided")
    
    # Resize all to same size
    max_width = max(img.width for img in images)
    max_height = max(img.height for img in images)
    
    resized = [
        img.resize((max_width, max_height), Image.Resampling.LANCZOS)
        for img in images
    ]
    
    rows = (len(images) + cols - 1) // cols
    
    grid_width = max_width * cols
    grid_height = max_height * rows
    
    grid = Image.new("RGB", (grid_width, grid_height), (255, 255, 255))
    
    for idx, img in enumerate(resized):
        row = idx // cols
        col = idx % cols
        x = col * max_width
        y = row * max_height
        grid.paste(img, (x, y))
    
    return grid
