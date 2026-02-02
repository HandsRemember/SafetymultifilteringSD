"""CoCa (Contrastive Captioners) for image-to-text captioning and embedding."""

import logging
from dataclasses import dataclass

import torch
import numpy as np
from PIL import Image
import open_clip

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class CoCaResult:
    """Result of CoCa analysis."""
    caption: str
    embedding: np.ndarray | None = None


class CoCaEmbedder:
    """CoCa model for image captioning and contrastive embedding.
    
    CoCa combines contrastive loss with captioning loss, enabling both
    image-text similarity computation and image-to-text generation.
    Reference: https://arxiv.org/abs/2205.01917
    """
    
    def __init__(
        self,
        model_name: str | None = None,
        pretrained: str | None = None,
        device: str | None = None,
    ):
        self.model_name = model_name or settings.models.coca_model
        self.pretrained = pretrained or settings.models.coca_pretrained
        self.device = device or settings.device
        self.model = None
        self.transform = None
        self._loaded = False
    
    def load(self) -> None:
        """Load CoCa model."""
        if self._loaded:
            return
        
        self.model, _, self.transform = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=self.pretrained,
            device=self.device,
        )
        self.model.eval()
        self._loaded = True
    
    def unload(self) -> None:
        """Unload model to free VRAM."""
        if self.model is not None:
            del self.model
            self.model = None
        self._loaded = False
        torch.cuda.empty_cache()
    
    def generate_caption(
        self,
        image: Image.Image,
        max_length: int = 30,
    ) -> str:
        """Generate caption for an image using CoCa's multimodal decoder."""
        if not self._loaded:
            self.load()
        
        image_input = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            generated = self.model.generate(
                image_input,
                seq_len=max_length,
            )
        
        caption = open_clip.decode(generated[0]).split("<end_of_text>")[0]
        caption = caption.replace("<start_of_text>", "").strip()
        
        return caption
    
    def get_image_embedding(self, image: Image.Image) -> np.ndarray:
        """Get normalized image embedding from CoCa's image encoder."""
        if not self._loaded:
            self.load()
        
        image_input = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu().numpy()[0]
    
    def get_text_embedding(self, text: str) -> np.ndarray:
        """Get normalized text embedding from CoCa's text encoder."""
        if not self._loaded:
            self.load()
        
        tokenizer = open_clip.get_tokenizer(self.model_name)
        text_tokens = tokenizer([text]).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features.cpu().numpy()[0]
    
    def analyze(
        self,
        image: Image.Image,
        include_embedding: bool = False,
    ) -> CoCaResult:
        """Analyze image: generate caption and optionally embedding."""
        caption = self.generate_caption(image)
        
        embedding = None
        if include_embedding:
            embedding = self.get_image_embedding(image)
        
        return CoCaResult(caption=caption, embedding=embedding)
    
    def compute_similarity(
        self,
        image: Image.Image,
        text: str,
    ) -> float:
        """Compute cosine similarity between image and text."""
        image_emb = self.get_image_embedding(image)
        text_emb = self.get_text_embedding(text)
        
        similarity = np.dot(image_emb, text_emb)
        return float(similarity)
