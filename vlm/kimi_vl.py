import requests
from typing import List, Dict, Tuple, Optional
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers.utils import is_flash_attn_2_available

# from vllm import LLM, SamplingParams


def extract_thinking_and_summary(text: str, bot: str = "◁think▷", eot: str = "◁/think▷") -> Tuple[str, str]:
    if bot in text and eot not in text:
        return "", text
    if eot in text:
        return text[text.index(bot) + len(bot):text.index(eot)].strip(), text[text.index(eot) + len(eot):].strip()
    return "", text


class KimiVL:
    def __init__(
        self,
        model=None,
        model_path: str = "moonshotai/Kimi-VL-A3B-Thinking-2506",
        use_vllm: bool = True,
        interleaved_visuals: bool = False,
    ) -> None:
        """Kimi-VL-A3B-Thinking model wrapper (API-compatible with Qwen2_5_VL)."""
        self.use_vllm = use_vllm
        self.interleaved_visuals = interleaved_visuals

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        if use_vllm:
            # Expect caller to pass a vLLM model instance
            self.model = model
            # self.sampling_params = SamplingParams(max_tokens=32768, temperature=0.8)
        else:
            # Standard Hugging Face Transformers path
            attn_implementation = "flash_attention_2" if is_flash_attn_2_available() else None
            print(f"Using {attn_implementation} for attention implementation")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype="auto",
                device_map="auto",
                attn_implementation=attn_implementation,
                trust_remote_code=True,
            )

    def __call__(
        self,
        inputs: List[Dict],
        sampling_params=None,
        generate_kwargs: Optional[dict] = None,
    ) -> Dict[str, str]:
        """
        Args:
            inputs (List[dict]): [
                {"type": "image", "content": "http://example.com/cat.jpg"},
                {"type": "text", "content": "What kind of cat is this? Answer with one word."}
            ]
        Returns:
            dict: {"thinking": "...", "summary": "..."}
        """
        if generate_kwargs is None:
            generate_kwargs = {}

        # Build conversation
        messages = [{"role": "user", "content": []}]
        images, image_placeholders = [], []

        for item in inputs:
            if item["type"] == "text":
                messages[0]["content"].append({"type": "text", "text": item["content"]})
            elif item["type"] == "image":
                if item["content"].startswith("http"):
                    img = Image.open(requests.get(item["content"], stream=True).raw).convert("RGB")
                else:
                    img = Image.open(item["content"]).convert("RGB")
                images.append(img)
                messages[0]["content"].append({"type": "image", "image": ""})
                image_placeholders.append(item["content"])

        # Apply chat template
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")

        if self.use_vllm:
            # vLLM inference
            outputs = self.model.generate(
                [{"prompt": text, "multi_modal_data": {"image": images}}],
                sampling_params=sampling_params,
                use_tqdm=False,
            )
            generated_text = outputs[0].outputs[0].text
        else:
            # Hugging Face Transformers inference
            inputs_tensor = self.processor(
                images=images, text=text, return_tensors="pt", padding=True, truncation=True
            ).to(self.model.device)

            generated_ids = self.model.generate(
                **inputs_tensor,
                max_new_tokens=4096,
                temperature=0.8,
            )

            # Trim prompt tokens from outputs
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs_tensor.input_ids, generated_ids)
            ]

            generated_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

        thinking, summary = extract_thinking_and_summary(generated_text)
        return {"thinking": thinking, "summary": summary}
