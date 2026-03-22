"""CLIP embedding baseline for safety checking (benchmark comparison)."""

import logging
from dataclasses import dataclass

import torch
import numpy as np
from PIL import Image
import open_clip

from config import settings

logger = logging.getLogger(__name__)

# Unsafe concept embeddings
# Kaynak: Rando et al. (2022) "Red-Teaming the Stable Diffusion Safety Filter"
# Table 1 — reverse-engineered SD v1.4 safety filter concepts (17 unsafe + special care)
# + ek kavramlar: gore, violence, bloody, disturbing (SD'nin kapsamadığı alanlar)
UNSAFE_CONCEPTS = [
    # --- Cinsel içerik (SD orijinal kavramlar, Table 1) ---
    "sexual",
    "nude",
    "sex",
    "naked",
    "nsfw",
    "porn",
    "18+",
    "explicit content",
    "uncensored",
    "nipples",
    "naked breast",
    "vagina",
    "areola",
    "naked person",
    # --- Şiddet / gore (SD'de yok — bu pipeline'ın ek katkısı) ---
    "gore",
    "violence",
    "bloody",
    "disturbing",
    # --- Çocuk koruması (SD special care concepts, Table 1) ---
    "young child",
    "young girl",
]


@dataclass
class CLIPCheckResult:
    """Result of CLIP-based safety check."""
    is_safe: bool
    max_similarity: float
    matched_concept: str | None
    all_similarities: dict[str, float]
    triggered_concepts: list[str] = None  # threshold'u geçen tüm class'lar

    def __post_init__(self):
        if self.triggered_concepts is None:
            self.triggered_concepts = []


class CLIPEmbedder:
    """CLIP-based embedding for safety checking (baseline method)."""
    
    def __init__(
        self,
        model_name: str | None = None,
        pretrained: str | None = None,
        device: str | None = None,
    ):
        self.model_name = model_name or settings.models.clip_model
        self.pretrained = pretrained or settings.models.clip_pretrained
        self.device = device or settings.device
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self._loaded = False
        self._concept_embeddings = None
    
    def load(self) -> None:
        """Load CLIP model."""
        if self._loaded:
            return
        
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=self.pretrained,
            device=self.device,
        )
        self.tokenizer = open_clip.get_tokenizer(self.model_name)
        self.model.eval()
        
        # Pre-compute unsafe concept embeddings
        self._compute_concept_embeddings()
        
        self._loaded = True
    
    def unload(self) -> None:
        """Unload model to free VRAM."""
        if self.model is not None:
            del self.model
            self.model = None
        self._concept_embeddings = None
        self._loaded = False
        torch.cuda.empty_cache()
    
    def _compute_concept_embeddings(self) -> None:
        """Pre-compute embeddings for unsafe concepts."""
        text_tokens = self.tokenizer(UNSAFE_CONCEPTS).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        self._concept_embeddings = text_features
    
    def get_image_embedding(self, image: Image.Image) -> np.ndarray:
        """Get normalized image embedding."""
        if not self._loaded:
            self.load()
        
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu().numpy()[0]
    
    def get_text_embedding(self, text: str) -> np.ndarray:
        """Get normalized text embedding."""
        if not self._loaded:
            self.load()
        
        text_tokens = self.tokenizer([text]).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features.cpu().numpy()[0]
    
    def compute_similarity(
        self,
        image: Image.Image,
        text: str,
    ) -> float:
        """Compute cosine similarity between image and text."""
        if not self._loaded:
            self.load()
        
        image_emb = self.get_image_embedding(image)
        text_emb = self.get_text_embedding(text)
        
        similarity = np.dot(image_emb, text_emb)
        return float(similarity)
    
    def check_safety(
        self,
        image: Image.Image,
        threshold: float | None = None,
    ) -> CLIPCheckResult:
        """Check image safety against unsafe concepts.

        Her kavram için settings.thresholds.clip_per_class_thresholds içindeki
        eşik kullanılır. Kavram orada tanımlanmamışsa global embedding_threshold
        devreye girer. threshold parametresi verilirse tüm class'lar için
        bu değer geçerli olur (per-class ayarları geçersiz kılınır).
        """
        if not self._loaded:
            self.load()

        global_threshold = threshold or settings.thresholds.embedding_threshold
        per_class = settings.thresholds.clip_per_class_thresholds

        # Get image embedding
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Compute similarities with all unsafe concepts
        similarities = (image_features @ self._concept_embeddings.T).squeeze(0)
        similarities = similarities.cpu().numpy()

        # Build results
        all_similarities = {
            concept: float(sim)
            for concept, sim in zip(UNSAFE_CONCEPTS, similarities)
        }

        max_idx = np.argmax(similarities)
        max_similarity = float(similarities[max_idx])

        # Her kavram için kendi threshold'unu kullan, yoksa global'e düş
        triggered = [
            UNSAFE_CONCEPTS[i]
            for i, sim in enumerate(similarities)
            if float(sim) > (
                per_class.get(UNSAFE_CONCEPTS[i], global_threshold)
                if threshold is None   # manuel threshold verilmemişse per-class geçerli
                else global_threshold
            )
        ]

        return CLIPCheckResult(
            is_safe=len(triggered) == 0,
            max_similarity=max_similarity,
            matched_concept=triggered[0] if triggered else None,
            all_similarities=all_similarities,
            triggered_concepts=triggered,
        )
