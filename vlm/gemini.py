import os
import io
import requests  # To handle image URLs
from typing import List, Dict, Any
from google import genai
from google.genai import types
from PIL import Image  # Used to load images


class Gemini:
    def __init__(self, model: str = "gemini-2.5-flash-lite"):
        """
        Google Gemini Vision model wrapper.
        
        Args:
            model (str): The name of the Gemini model to use 
                         (e.g., "gemini-1.5-flash-latest").
        """
        try:
            self.model_name = model
            self.model = genai.Client() #genai.GenerativeModel(model_name=model)
            print(f"Gemini model '{model}' initialized.")
        except Exception as e:
            print(f"Error initializing Gemini model '{model}': {e}")
            self.model = None

    def __call__(self, inputs: List[Dict[str, str]], 
                 generate_kwargs: Dict[str, Any] = None) -> str:
        """
        Makes a multimodal call to the Gemini API.

        Args:
            inputs (List[dict]): A list of dictionaries, each specifying
                                 a "type" ("text" or "image") and "content".
                Example: [
                    {"type": "image", "content": "path_or_url_to_image.jpg"},
                    {"type": "text", "content": "What is in the picture?"}
                ]
            
            generate_kwargs (dict, optional): A dictionary of generation parameters
                to pass to Gemini, such as "temperature", "max_output_tokens",
                "top_p", or "top_k".

        Returns:
            str: The generated text response from the model.
        """
        if self.model is None:
            return "[Error: Gemini model not initialized. Check API key and configuration.]"

        # Build the content payload for Gemini
        # Gemini accepts a list of parts, which can be text strings or PIL Images.
        content_parts = []
        for item in inputs:
            item_type = item.get("type")
            item_content = item.get("content")

            if not item_content:
                print(f"Warning: Skipping item with no content: {item}")
                continue

            if item_type == "text":
                content_parts.append(item_content)
                
            elif item_type == "image":
                try:
                    if item_content.startswith("http://") or item_content.startswith("https://"):
                        # Handle image from URL
                        response = requests.get(item_content)
                        response.raise_for_status()  # Raise an error for bad status codes
                        img = Image.open(io.BytesIO(response.content))
                    else:
                        # Handle image from local file path
                        img = Image.open(item_content)
                    content_parts.append(img)
                except requests.exceptions.RequestException as e:
                    print(f"Warning: Could not fetch image from URL {item_content}. Error: {e}")
                    content_parts.append(f"[Image at {item_content} could not be loaded]")
                except FileNotFoundError:
                    print(f"Warning: Image file not found at {item_content}.")
                    content_parts.append(f"[Image file at {item_content} not found]")
                except Exception as e:
                    print(f"Warning: Could not load image {item_content}. Error: {e}")
                    content_parts.append(f"[Image {item_content} could not be loaded]")
            else:
                print(f"Warning: Unknown input type '{item_type}'. Skipping.")

        if not content_parts:
            return "[Error: No valid content parts to send to API.]"

        # Map generate_kwargs to Gemini's GenerationConfig
        # The keys (temperature, max_output_tokens, top_p, top_k)
        # are directly compatible.
        #generate_kwargs['thinking_budget'] = 0

        # Call Gemini API
        try:
            response = self.model.models.generate_content(
                model=self.model_name,
                contents=content_parts,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=0)
                    # Turn off thinking:
                    # thinking_config=types.ThinkingConfig(thinking_budget=0)
                    # Turn on dynamic thinking:
                    # thinking_config=types.ThinkingConfig(thinking_budget=-1)
                ),
            )
            
            # Access the text response
            return response.text.strip()
        except Exception as e:
            # Handle potential API errors (e.g., safety settings, bad request)
            print(f"Error calling Gemini API: {e}")
            return f"[Gemini API Error: {e}]"