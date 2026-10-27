"""Frozen DINOv2 CLS, UMT5 token features, and CLIP action-text features."""
import torch
from PIL import Image
from transformers import (AutoImageProcessor, AutoTokenizer, CLIPTextModelWithProjection,
                          Dinov2Model, UMT5EncoderModel)


class FrozenEncoders:
    def __init__(self, modality="text", visual_model="facebook/dinov2-base",
                 condition_model=None, tokenizer=None, device="cpu", max_condition_tokens=128):
        if modality not in ("text", "action"):
            raise ValueError("unknown modality")
        self.modality, self.device = modality, torch.device(device)
        self.max_condition_tokens = max_condition_tokens
        self.visual_source = str(visual_model)
        self.condition_source = str(condition_model or (
            "google/umt5-xxl" if modality == "text" else "openai/clip-vit-base-patch32"))
        self.image_processor = AutoImageProcessor.from_pretrained(visual_model, use_fast=False)
        self.visual = Dinov2Model.from_pretrained(visual_model).to(self.device).eval()
        if self.visual.config.hidden_size != 768:
            raise ValueError("paper requires DINOv2 ViT-B/14, feature dimension 768")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer or self.condition_source)
        if modality == "text":
            self.condition = UMT5EncoderModel.from_pretrained(
                self.condition_source, torch_dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
            ).to(self.device).eval()
            if self.condition.config.d_model != 4096:
                raise ValueError("text selector requires UMT5-XXL (4096 dimensions)")
        else:
            self.condition = CLIPTextModelWithProjection.from_pretrained(self.condition_source).to(self.device).eval()
            if self.condition.config.projection_dim != 512:
                raise ValueError("action selector requires CLIP ViT-B/32 (512 dimensions)")
        self.visual.requires_grad_(False)
        self.condition.requires_grad_(False)

    @torch.inference_mode()
    def images(self, paths, batch_size=16):
        if not paths:
            return torch.empty(0, 768)
        features = []
        for start in range(0, len(paths), batch_size):
            images = []
            for path in paths[start:start + batch_size]:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            inputs = self.image_processor(images=images, return_tensors="pt").to(self.device)
            # The final CLS token, without teacher normalization on student inputs.
            features.append(self.visual(**inputs).last_hidden_state[:, 0].float().cpu())
        return torch.cat(features)

    @torch.inference_mode()
    def text(self, prompt):
        return self.text_many([prompt])[0]

    @torch.inference_mode()
    def text_many(self, prompts):
        if not prompts or any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompts):
            raise ValueError("known rollout condition must be nonempty text")
        limit = min(self.max_condition_tokens, 77) if self.modality == "action" else self.max_condition_tokens
        inputs = self.tokenizer(prompts, padding=True, return_tensors="pt", truncation=False).to(self.device)
        if inputs.input_ids.shape[1] > limit:
            raise ValueError(f"condition exceeds {limit} tokens; shorten explicitly rather than silently truncating")
        outputs = self.condition(**inputs)
        if self.modality == "action":
            return [embedding.float().cpu().unsqueeze(0) for embedding in outputs.text_embeds]
        return [embedding[mask.bool()].float().cpu() for embedding, mask in zip(outputs.last_hidden_state, inputs.attention_mask)]

    def metadata(self):
        return {"modality": self.modality, "visual_model": self.visual_source,
                "condition_model": self.condition_source, "max_condition_tokens": self.max_condition_tokens}
