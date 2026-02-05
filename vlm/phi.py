import torch
from typing import List
from PIL import Image
import requests

from transformers import AutoProcessor, AutoModelForCausalLM
from transformers.utils import is_flash_attn_2_available


class PhiVision:
    def __init__(self, model_path: str = "microsoft/Phi-3.5-vision-instruct") -> None:
        """Phi-3.5-Vision model wrapper (API-compatible with Qwen2_5_VL style)."""
        attn_impl = "flash_attention_2" if is_flash_attn_2_available() else "eager"
        print(f"Using {attn_impl} for attention implementation")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="cuda",
            trust_remote_code=True,
            torch_dtype="auto",
            _attn_implementation=attn_impl,
        )

        # For best performance: num_crops=4 for multi-frame, 16 for single-frame
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            num_crops=1,
        )

    def __call__(self, inputs: List[dict], generate_kwargs=None) -> str:
        """
        Args:
            inputs (List[dict]): [
                {"type": "image", "content": "http://example.com/img1.jpg"},
                {"type": "text", "content": "What is in the picture?"}
            ]
        """
        if generate_kwargs is None:
            generate_kwargs = {"max_new_tokens": 200, "temperature": 0.0, "do_sample": False}

        images = []
        placeholder = ""
        conversation = [{"role": "user", "content": ""}]

        for idx, item in enumerate(inputs, 1):
            if item["type"] == "text":
                conversation[0]["content"] += item["content"]
            elif item["type"] == "image":
                if item["content"].startswith("http"):
                    img = Image.open(requests.get(item["content"], stream=True).raw).convert("RGB")
                else:
                    img = Image.open(item["content"]).convert("RGB")
                images.append(img)
                placeholder += f"<|image_{idx}|>\n"

        # Append placeholder if images exist
        if images:
            conversation[0]["content"] = placeholder + conversation[0]["content"]

        # Apply Phi chat template
        prompt = self.processor.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )

        # Preprocess
        proc_inputs = self.processor(prompt, images, return_tensors="pt").to("cuda:0")

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **proc_inputs,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                **generate_kwargs
            )

        # Remove input tokens
        output_ids = output_ids[:, proc_inputs["input_ids"].shape[1]:]

        return self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0].strip()
