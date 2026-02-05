import os
import torch
import time
from typing import List
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLMoeForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from transformers.image_utils import load_image
from transformers.utils import is_flash_attn_2_available
#from vllm import LLM, SamplingParams

class Qwen3_VL():
    def __init__(self, model=None, model_path:str="Qwen/Qwen3-VL-8B-Instruct", use_vllm=True, interleaved_visuals=False) -> None:
        """Llava model wrapper

        Args:
            model_path (str): model name
        """
        attn_implementation = "flash_attention_2" if is_flash_attn_2_available() else None
        print(f"Using {attn_implementation} for attention implementation")
        self.use_vllm = use_vllm
        if use_vllm:
            self.model = model
        else:
            if 'a3b' in model_path.lower():
                self.model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    attn_implementation=attn_implementation
                )
            else:
                self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    attn_implementation=attn_implementation
                )
        min_pixels = 16 * 28 * 28
        max_pixels = 2048 * 28 * 28
        self.processor = AutoProcessor.from_pretrained(
            model_path, min_pixels=min_pixels, max_pixels=max_pixels
        )
        self.interleaved_visuals = interleaved_visuals

        
    def __call__(self, inputs: List[dict], sampling_params=None, generate_kwargs={}) -> str:
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
                    user_content.append({"type": "image", "image": load_image(item['content'])})
                elif item["type"] == "text":
                    user_content.append({"type": "text", "text": item["content"]})
            messages.append({"role": "user", "content": user_content})
        else:
            # non-interleaved: all images first, then text
            text_prompt = "\n".join([x["content"] for x in inputs if x["type"] == "text"])
            messages = [
            {
                "role": "user",
                "content": [ {"type": "image", "image": load_image(img)} for img in image_links] + [{"type": "text", "text": text_prompt}]
            }
            ]
            
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        #image_inputs, video_inputs = process_vision_info(messages)
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True)

        if self.use_vllm:
            # Create vLLM input
            mm_data = {}
            if image_inputs is not None:
                mm_data["image"] = image_inputs
            
            llm_input = {
                "prompt": text,
                "multi_modal_data": mm_data,
            }
            outputs = self.model.generate([llm_input], sampling_params=sampling_params, use_tqdm=False)
            pred_answer = outputs[0].outputs[0].text
        else:
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.model.device)

            # === Inference ===
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs, 
                    **generate_kwargs
                )
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_texts = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )
            pred_answer = output_texts[0].strip()
        
        return pred_answer