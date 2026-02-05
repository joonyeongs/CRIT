import torch
from typing import List
from PIL import Image
import requests

from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
from transformers.utils import is_flash_attn_2_available


class LLaVAOneVision:
    def __init__(self, model=None, model_path: str = "llava-hf/llava-onevision-qwen2-7b-ov-hf", use_vllm: bool = True, interleaved_visuals: bool = False) -> None:
        """LLaVA-OneVision model wrapper (API-compatible with Qwen2_5_VL)."""
        attn_implementation = "flash_attention_2" if is_flash_attn_2_available() else None
        print(f"Using {attn_implementation} for attention implementation")

        self.use_vllm = use_vllm
        self.interleaved_visuals = interleaved_visuals

        if use_vllm:
            # vLLM support not fully available yet for LLaVA-OneVision
            # keep placeholder to preserve init signature
            from vllm import SamplingParams
            self.sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=1024,
            )
            self.model = model
        else:
            self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                attn_implementation=attn_implementation,
            ).to("cuda")

        self.processor = AutoProcessor.from_pretrained(model_path)

    def __call__(self, inputs: List[dict], generate_kwargs=None) -> str:
        """
        Args:
            inputs (List[dict]): [
                {"type": "image", "content": "http://example.com/img1.jpg"},
                {"type": "text", "content": "What is in the picture?"}
            ]
        """
        if generate_kwargs is None:
            generate_kwargs = {"max_new_tokens": 200, "do_sample": False}

        # Build conversation
        conversation = [{"role": "user", "content": []}]
        images = []

        for item in inputs:
            if item["type"] == "text":
                conversation[0]["content"].append({"type": "text", "text": item["content"]})
            elif item["type"] == "image":
                if item["content"].startswith("http"):
                    img = Image.open(requests.get(item["content"], stream=True).raw).convert("RGB")
                else:
                    img = Image.open(item["content"]).convert("RGB")
                conversation[0]["content"].append({"type": "image"})
                images.append(img)

        # Apply chat template
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

        if self.use_vllm:
            # Placeholder: vLLM integration would require custom adaptation
            raise NotImplementedError("vLLM support for LLaVA-OneVision is not implemented yet.")
        else:
            # Preprocess
            proc_inputs = self.processor(images=images, text=prompt, return_tensors="pt").to(self.model.device, torch.float16)

            # Generate
            with torch.no_grad():
                output = self.model.generate(**proc_inputs, **generate_kwargs)

            return self.processor.decode(
                output[0][proc_inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            ).strip()
