import base64
from typing import List
from openai import OpenAI

class OpenAIVision:
    def __init__(self, model: str = "gpt-4o"):
        """OpenAI Vision model wrapper (API-compatible with Qwen2_5_VL/LLaVA)."""
        self.client = OpenAI()
        self.model = model

    def encode_image(self, image_path: str) -> str:
        """Convert image to base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def __call__(self, inputs: List[dict], generate_kwargs=None) -> str:
        """
        Args:
            inputs (List[dict]): [
                {"type": "image", "content": "path_or_url_to_image.jpg"},
                {"type": "text", "content": "What is in the picture?"}
            ]
        """
        # Build conversation payload
        content_blocks = []
        for item in inputs:
            if item["type"] == "text":
                content_blocks.append({"type": "input_text", "text": item["content"]})
            elif item["type"] == "image":
                if item["content"].startswith("http"):
                    # Direct URL
                    content_blocks.append({
                        "type": "input_image",
                        "image_url": item["content"],
                        "detail": "low"
                    })
                else:
                    # Local file -> base64 encode
                    base64_img = self.encode_image(item["content"])
                    content_blocks.append({
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{base64_img}",
                        "detail": "low"
                    })

        # Call OpenAI Responses API
        generate_kwargs={"temperature": 0.0, "max_output_tokens": 1024}
        # response = self.client.responses.create(
        #     model="gpt-4o",
        #     input=[{"role": "user", "content": content_blocks}],
        #     **(generate_kwargs or {})   # 👈 your control here
        # )
        response = self.client.responses.create(
            model="gpt-5",
            reasoning={"effort": "low"},
            input=[{"role": "user", "content": content_blocks}],
        )   # 👈 your control here

        return response.output_text.strip()
