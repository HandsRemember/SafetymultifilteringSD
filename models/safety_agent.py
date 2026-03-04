"""Pre-generation Safety Agent using LLM for prompt analysis."""

import logging
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class SafetyCheckResult:
    """Result of safety check."""
    is_safe: bool
    reason: str
    category: str | None = None
    confidence: float = 1.0


SAFETY_SYSTEM_PROMPT = """You are a content safety classifier. Analyze prompts for inappropriate content.

Unsafe categories: sexual, violence, hate, harassment, self-harm, illegal, shocking.

IMPORTANT: You MUST respond with ONLY a valid JSON object, nothing else.
Format: {"safe": true, "reason": "explanation", "category": null}
Or if unsafe: {"safe": false, "reason": "explanation", "category": "category_name"}

Examples:
- "A sunset over ocean" → {"safe": true, "reason": "Nature scene", "category": null}
- "Nude woman" → {"safe": false, "reason": "Explicit nudity request", "category": "sexual"}
- "A girl running in a field" → {"safe": true, "reason": "Innocent activity", "category": null}
- "A person holding a razor blade to their wrist" → {"safe": false, "reason": "Violent content", "category": "self-harm"}
- "A photorealistic close-up of a sword piercing a soldier's chest with blood splashing" → {"safe": false, "reason": "Violent content", "category": "violence"}

Avoid false positives. Normal people, activities, and scenes are SAFE."""


class SafetyAgent:
    """LLM-based pre-generation safety checker."""
    
    def __init__(self, model_id: str | None = None, device: str | None = None):
        self.model_id = model_id or settings.models.safety_agent_id
        self.device = device or settings.device
        self.model = None
        self.tokenizer = None
        self._loaded = False
    
    def load(self) -> None:
        """Load the model with quantization."""
        if self._loaded:
            return
            
        quantization = settings.models.safety_agent_quantization
        
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
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        
        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        
        self._loaded = True
    
    def unload(self) -> None:
        """Unload model to free VRAM."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self._loaded = False
        torch.cuda.empty_cache()
    
    def check_prompt(self, prompt: str) -> SafetyCheckResult:
        """Check if a prompt is safe for image generation."""
        if not self._loaded:
            self.load()
        
        messages = [
            {"role": "system", "content": SAFETY_SYSTEM_PROMPT},
            {"role": "user", "content": f"Analyze this prompt: {prompt}"},
        ]
        
        inputs = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
            return_dict=True,
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        input_length = input_ids.shape[1]
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(
            outputs[0][input_length:],
            skip_special_tokens=True,
        )
        
        return self._parse_response(response)
    
    def _parse_response(self, response: str) -> SafetyCheckResult:
        """Parse LLM response into SafetyCheckResult."""
        import json
        import re
        
        # Try JSON parsing first
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start != -1 and end > start:
                json_str = response[start:end]
                data = json.loads(json_str)
                
                return SafetyCheckResult(
                    is_safe=data.get("safe", False),
                    reason=data.get("reason", "Unknown"),
                    category=data.get("category"),
                )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"JSON parse failed, using fallback: {e}")
        
        # Fallback: Text-based analysis
        response_lower = response.lower()
        
        # Check for explicit unsafe indicators
        unsafe_indicators = [
            "unsafe", "inappropriate", "explicit", "nudity", "sexual",
            "violence", "harmful", "not safe", "cannot generate", "refuse"
        ]
        
        # Check for explicit safe indicators
        safe_indicators = [
            "safe", "appropriate", "innocent", "benign", "acceptable",
            "no inappropriate", "nothing inappropriate", "does not contain"
        ]
        
        # Count matches
        unsafe_count = sum(1 for ind in unsafe_indicators if ind in response_lower)
        safe_count = sum(1 for ind in safe_indicators if ind in response_lower)
        
        # Decide based on indicator balance
        if safe_count > unsafe_count:
            return SafetyCheckResult(
                is_safe=True,
                reason=response[:100],
                category=None,
            )
        elif unsafe_count > 0:
            return SafetyCheckResult(
                is_safe=False,
                reason=response[:100],
                category="unknown",
            )
        
        # If unclear, default to SAFE (reduce false positives)
        return SafetyCheckResult(
            is_safe=True,
            reason=f"Fallback: {response[:80]}",
            category=None,
        )
    
    def check_caption(self, caption: str) -> SafetyCheckResult:
        """Re-check generated image caption for safety."""
        return self.check_prompt(caption)
