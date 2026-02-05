import requests
from typing import List
from PIL import Image

import torch
from transformers import MllamaForConditionalGeneration, AutoProcessor


class LlamaVision:
    def __init__(
        self,
        model=None,
        model_path: str = "meta-llama/Llama-3.2-11B-Vision",
        use_vllm: bool = True,
        interleaved_visuals: bool = False,
    ) -> None:
        """Llama-3.2 Vision wrapper (API-compatible with Qwen2_5_VL)."""
        self.use_vllm = use_vllm
        self.interleaved_visuals = interleaved_visuals

        if use_vllm:
            # vLLM not yet officially supporting Llama-Vision
            raise NotImplementedError("vLLM support for Llama-3.2-Vision is not available yet.")
        else:
            self.model = MllamaForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        self.processor = AutoProcessor.from_pretrained(model_path)

    def __call__(self, inputs: List[dict], generate_kwargs=None) -> str:
        """
        Args:
            inputs (List[dict]): [
                {"type": "image", "content": "http://example.com/img.jpg"},
                {"type": "text", "content": "Write a haiku about this."}
            ]
        Returns:
            str: model-generated text
        """
        if generate_kwargs is None:
            generate_kwargs = {"max_new_tokens": 1024}

        images = []
        text_prompt_parts = []

        # Gather inputs
        for item in inputs:
            if item["type"] == "text":
                text_prompt_parts.append(item["content"])
            elif item["type"] == "image":
                if item["content"].startswith("http"):
                    img = Image.open(requests.get(item["content"], stream=True).raw).convert("RGB")
                else:
                    img = Image.open(item["content"]).convert("RGB")
                images.append(img)

        # Default: all images first, then text (non-interleaved mode)
        if self.interleaved_visuals:
            # Interleaved: prepend <|image|> tokens inline
            prompt = ""
            for item in inputs:
                if item["type"] == "image":
                    prompt += "<|image|>"
                elif item["type"] == "text":
                    prompt += item["content"]
        else:
            prompt = "".join(["<|image|>" for _ in images])
            if text_prompt_parts:
                prompt += "<|begin_of_text|>" + "\n".join(text_prompt_parts)

        # Preprocess
        proc_inputs = self.processor(images=images, text=prompt, return_tensors="pt").to(self.model.device)

        # Generate
        with torch.no_grad():
            output = self.model.generate(**proc_inputs, **generate_kwargs)

        return self.processor.decode(output[0], skip_special_tokens=True).strip()
