"""Vision Language Model checker for semantic visual safety analysis."""

import logging
from dataclasses import dataclass

import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig

from config import settings

logger = logging.getLogger(__name__)

# Supported model families and their transformers classes
_MODEL_FAMILIES = {
    "qwen": "Qwen2VLForConditionalGeneration",
    "llava": "LlavaNextForConditionalGeneration",
}

VLM_SAFETY_PROMPT = """You are a strict image safety classifier for a content moderation system.

Analyze this image carefully and check for ANY of these unsafe categories:

1. VIOLENCE: weapons (guns, knives, swords), fighting, blood, injuries, wounds, gore, dead bodies, torture, war scenes, explosions with victims
2. SELF_HARM: cutting, self-injury, suicide imagery, nooses, razor blades on skin, visible self-inflicted wounds
3. ILLEGAL_ACTIVITY: drug use/production, vandalism, theft, terrorism symbols, bomb-making
4. GORE: exposed organs, dismemberment, extreme body horror, mutilation
5. HATE: nazi symbols, swastikas, KKK imagery, racial caricatures, hate group symbols
6. HARASSMENT: threatening gestures, bullying scenes, intimidation
7. SHOCKING: extreme body modification, disturbing surreal imagery designed to disturb

Be STRICT. If there is ANY doubt, mark as unsafe.
Nudity is handled by another system - focus on the categories above.

You MUST respond with ONLY this JSON, no other text:
{"safe": true, "reason": "brief description of the image", "categories": []}
{"safe": false, "reason": "image shows weapons and blood", "categories": ["violence", "self_harm"]}"""

@dataclass
class VLMCheckResult:
    """Result of VLM safety check."""
    is_safe: bool
    reason: str
    categories: list[str]
    confidence: float = 1.0


def _detect_family(model_id: str) -> str:
    """Detect model family from model ID."""
    model_lower = model_id.lower()
    for family in _MODEL_FAMILIES:
        if family in model_lower:
            return family
    raise ValueError(
        f"Unsupported VLM: {model_id}. "
        f"Supported families: {list(_MODEL_FAMILIES.keys())}"
    )


def _load_model_class(family: str):
    """Import and return the correct model class for the family."""
    import transformers
    class_name = _MODEL_FAMILIES[family]
    return getattr(transformers, class_name)


class VLMChecker:
    """Vision Language Model for semantic visual safety analysis.

    Supports multiple model families (Qwen2-VL, LLaVA-Next).
    Model family is auto-detected from the model ID in config.
    """

    def __init__(self, model_id: str | None = None, device: str | None = None):
        self.model_id = model_id or settings.models.vlm_id
        self.device = device or settings.device
        self._family = _detect_family(self.model_id)
        self.model = None
        self.processor = None
        self._loaded = False

    def load(self) -> None:
        """Load VLM with quantization."""
        if self._loaded:
            return

        quantization = settings.models.vlm_quantization

        if quantization == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        elif quantization == "8bit":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            bnb_config = None

        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )

        model_class = _load_model_class(self._family)
        self.model = model_class.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )

        self._loaded = True

    def unload(self) -> None:
        """Unload model to free VRAM."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        self._loaded = False
        torch.cuda.empty_cache()

    def _build_messages(self, image: Image.Image, prompt: str) -> list[dict]:
        """Build chat messages in the correct format for the model family."""
        if self._family == "qwen":
            return [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]}]

        if self._family == "llava":
            return [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ]}]

        raise ValueError(f"Unknown family: {self._family}")

    def analyze(
        self,
        image: Image.Image,
        custom_prompt: str | None = None,
    ) -> VLMCheckResult:
        """Analyze image for safety using VLM."""
        if not self._loaded:
            self.load()

        prompt = custom_prompt or VLM_SAFETY_PROMPT
        messages = self._build_messages(image, prompt)

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
        )[0]

        return self._parse_response(response)
    
    def _parse_response(self, response: str) -> VLMCheckResult:
        """Parse VLM response into VLMCheckResult."""
        import json
        
        # Clean markdown code fences if present
        cleaned = response.replace("```json", "").replace("```", "").strip()
        
        # Try JSON parsing first
        try:
            start = cleaned.find("{")
            end = cleaned.rfind("}") + 1
            if start != -1 and end > start:
                json_str = cleaned[start:end]
                data = json.loads(json_str)
                
                return VLMCheckResult(
                    is_safe=data.get("safe", False),
                    reason=data.get("reason", "Unknown"),
                    categories=data.get("categories", []),
                )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"JSON parse failed, using fallback: {e}")
        
        # Fallback: Text-based analysis for VLM task categories
        response_lower = cleaned.lower()
        
        unsafe_words = ["violence", "blood", "gore", "weapon", "injury",
                       "hate", "racist", "discriminat", "self-harm", "self_harm",
                       "suicide", "cutting", "harassment", "threaten", "bully",
                       "unsafe", "not safe"]
        safe_words = ["safe", "no issues", "no violence", "no hate",
                     "clean", "appropriate", "nothing unsafe"]
        
        unsafe_count = sum(1 for w in unsafe_words if w in response_lower)
        safe_count = sum(1 for w in safe_words if w in response_lower)
        
        # Detect specific categories from response
        detected_categories = []
        category_keywords = {
            "violence": ["violence", "blood", "gore", "weapon", "injury", "fighting"],
            "hate": ["hate", "racist", "discriminat", "swastika"],
            "self_harm": ["self-harm", "self_harm", "suicide", "cutting"],
            "harassment": ["harassment", "threaten", "bully", "intimidat"],
        }
        for cat, keywords in category_keywords.items():
            if any(kw in response_lower for kw in keywords):
                detected_categories.append(cat)
        
        if safe_count > unsafe_count:
            return VLMCheckResult(
                is_safe=True,
                reason=response[:100],
                categories=[],
            )
        elif unsafe_count > 0:
            return VLMCheckResult(
                is_safe=False,
                reason=response[:100],
                categories=detected_categories or ["detected"],
            )
        
        # Default to safe if unclear
        return VLMCheckResult(
            is_safe=True,
            reason=f"Fallback: {response[:80]}",
            categories=[],
        )
