"""Vision Language Model checker for semantic visual safety analysis."""

import logging
from dataclasses import dataclass

import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

from config import settings

logger = logging.getLogger(__name__)


VLM_SAFETY_PROMPT = """You are an image safety classifier. Ignore nudity (handled separately).

Check ONLY for:
1. violence - fighting, weapons, blood, injuries, gore
2. hate - hate symbols, racist imagery, discriminatory content
3. self_harm - cutting, suicide, self-injury
4. harassment - threatening, bullying, intimidation

Answer with ONLY this JSON format, no other text:
{"safe": true, "reason": "no issues", "categories": []}

If unsafe, list ALL matching categories:
{"safe": false, "reason": "what you see", "categories": ["violence", "hate"]}"""

@dataclass
class VLMCheckResult:
    """Result of VLM safety check."""
    is_safe: bool
    reason: str
    categories: list[str]
    confidence: float = 1.0


class VLMChecker:
    """Vision Language Model for semantic visual safety analysis.
    
    Uses Qwen2-VL-7B to analyze images for complex safety violations
    that may not be detected by specialized classifiers like NudeNet.
    """
    
    def __init__(self, model_id: str | None = None, device: str | None = None):
        self.model_id = model_id or settings.models.vlm_id
        self.device = device or settings.device
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
        
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
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
    
    def analyze(
        self,
        image: Image.Image,
        custom_prompt: str | None = None,
    ) -> VLMCheckResult:
        """Analyze image for safety using VLM."""
        if not self._loaded:
            self.load()
        
        prompt = custom_prompt or VLM_SAFETY_PROMPT
        
        # Prepare messages for Qwen2-VL
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        # Process inputs
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
        
        # Generate response
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
            )
        
        # Decode response
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
