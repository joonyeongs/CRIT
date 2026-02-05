import torch
from typing import List
from PIL import Image
import requests
from transformers import AutoModel, AutoTokenizer

# ===== InternVL image preprocessing utils =====
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def build_transform(input_size=448):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_image(image_file, input_size=448, max_num=12):
    #from internvl_utils import dynamic_preprocess  # assume you move dynamic_preprocess into its own file or above
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


# ===== Wrapper Class =====
class InternVL:
    def __init__(self, model_path: str = "OpenGVLab/InternVL2_5-8B") -> None:
        """InternVL wrapper (API-compatible with LLaVAOneVision)."""

        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True
        ).eval().cuda()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=False
        )

        self.generation_config = dict(max_new_tokens=512, do_sample=False)

    def __call__(self, inputs: List[dict], generate_kwargs=None) -> str:
        """
        Args:
            inputs (List[dict]): [
                {"type": "image", "content": "http://example.com/img1.jpg"},
                {"type": "text", "content": "What is in the picture?"}
            ]
        """
        if generate_kwargs is None:
            generate_kwargs = self.generation_config

        text_parts = []
        pixel_values_list = []
        num_patches_list = []

        # Build text & load images
        for item in inputs:
            if item["type"] == "text":
                text_parts.append(item["content"])
            elif item["type"] == "image":
                if item["content"].startswith("http"):
                    img = Image.open(requests.get(item["content"], stream=True).raw).convert("RGB")
                    tmp_path = "/tmp/tmp_internvl.jpg"
                    img.save(tmp_path)
                    pixel_values = load_image(tmp_path).to(torch.bfloat16).cuda()
                else:
                    pixel_values = load_image(item["content"]).to(torch.bfloat16).cuda()

                pixel_values_list.append(pixel_values)
                num_patches_list.append(pixel_values.size(0))
                text_parts.append("<image>")

        question = "\n".join(text_parts)

        # Concatenate multiple images if needed
        if pixel_values_list:
            pixel_values = torch.cat(pixel_values_list, dim=0)
        else:
            pixel_values = None

        # Call InternVL chat
        if pixel_values is not None:
            if len(num_patches_list) > 1:
                response, _ = self.model.chat(
                    self.tokenizer,
                    pixel_values,
                    question,
                    generate_kwargs,
                    num_patches_list=num_patches_list,
                    history=None,
                    return_history=True
                )
            else:
                response = self.model.chat(
                    self.tokenizer,
                    pixel_values,
                    question,
                    generate_kwargs
                )
        else:
            response, _ = self.model.chat(
                self.tokenizer,
                None,
                question,
                generate_kwargs,
                history=None,
                return_history=True
            )

        return response.strip()
