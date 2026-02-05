import os
import torch
import time
from typing import List
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image
from transformers.utils import is_flash_attn_2_available


class Idefics2():
    def __init__(self, model_path:str="HuggingFaceM4/idefics2-8b", interleaved_visuals=False) -> None:
        """Llava model wrapper

        Args:
            model_path (str): model name
        """
        attn_implementation = "flash_attention_2" if is_flash_attn_2_available() else None
        print(f"Using {attn_implementation} for attention implementation")
        self.model = AutoModelForVision2Seq.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16, _attn_implementation=attn_implementation).eval()
        self.processor = AutoProcessor.from_pretrained(model_path, do_image_splitting=False)
        self.interleaved_visuals = interleaved_visuals

        
    def __call__(self, inputs: List[dict], generate_kwargs={}) -> str:
        """
        Args:
            inputs (List[dict]): [
                {
                    "type": "image",
                    "content": "https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/input/sample_34_1.jpg"
                },
                {
                    "type": "image",
                    "content": "https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/input/sample_337180_3.jpg"
                },
                {
                    "type": "text",
                    "content": "What is difference between two images?"
                }
            ]
            Supports any form of interleaved format of image and text.
        """
        image_links = [x["content"] for x in inputs if x["type"] == "image"]
        if self.interleaved_visuals:
            # Support interleaved format of image and text
            messages = []
            user_content = []
            for item in inputs:
                if item["type"] == "image":
                    user_content.append({"type": "image"})
                elif item["type"] == "text":
                    user_content.append({"type": "text", "text": item["content"]})
            messages.append({"role": "user", "content": user_content})
        else:
            # non-interleaved: all images first, then text
            text_prompt = "\n".join([x["content"] for x in inputs if x["type"] == "text"])
            messages = [
            {
                "role": "user",
                "content": [ {"type": "image"}] * len(image_links) + [{"type": "text", "text": text_prompt}]
            }
            ]
            
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        images = [load_image(image_link) for image_link in image_links]
        # Temp: text-only generation for debugging
        images = None
        proc_inputs = self.processor(text=prompt, images=images, return_tensors="pt")
        proc_inputs = {k: v.to(self.model.device) for k, v in proc_inputs.items()}
        generate_ids = self.model.generate(**proc_inputs, **generate_kwargs)
        generated_text = self.processor.batch_decode(generate_ids[:, proc_inputs['input_ids'].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return generated_text