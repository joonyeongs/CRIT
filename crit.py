import os
import pandas as pd
import json

class CRIT():
    def __init__(self, use_cot=True) -> None:
        if use_cot:
            self.task_prompt = "Based on the provided images and text context, answer the question step by step. After explaining your reasoning, output the final answer in the format 'Final Answer: <your answer>' where <your answer> is a single word or phrase."
        else:
            self.task_prompt = "Based on the provided images and text context, answer the question using a single word or phrase."

        self.natural_image_path = "natural_image_benchmark_total_for_eval_refined_wo_cross_image.json"
        self.video_path = "video_benchmark_total_for_eval_refined.json"
        self.scientific_paper_path = "scientific_paper_benchmark_total_for_eval_refined.json"
        
        self.dataset_split = {
            "natural_image": [],
            "video": [],
            "scientific_paper": []
        }       
        

        with open(self.natural_image_path, 'r') as f:
            self.dataset_split["natural_image"] = json.load(f)
        with open(self.video_path, 'r') as f:
            self.dataset_split["video"] = json.load(f)
        with open(self.scientific_paper_path, 'r') as f:
            self.dataset_split["scientific_paper"] = json.load(f)
        self.image_dir = {
            "natural_image": "data",
            "video": "data/ActivityNet-Captions",
            "scientific_paper": "data/spiqa/train_val"
        }

    def get_samples_by_split(self):
        converted_samples = {
            "natural_image": [],
            "video": [],
            "scientific_paper": []
        }
        print(f"Mode: {self.mode}")
        for split, data in self.dataset_split.items():
            for item in data:
                context = item["context"]
                images = item["images"]
                image_labels = item["image_labels"]
                question = item["question"]
                answer = item['answer'].strip()
                if '(' in answer and ')' in answer:
                    gt = answer.split('(')[0].strip()
                else:
                    gt = answer
                image_num_per_bin = item["image_num_per_bin"]
                image_to_find = item["image_to_find"]

                image_text_sequence = []
                image_counter = 0
                texts = []
                for i, image_num in enumerate(image_num_per_bin):
                    if image_num > 0:
                        for _ in range(image_num):
                            if split == "scientific_paper":
                                image_text_sequence.append('text')
                                texts.append(image_labels[image_counter])
                            image_text_sequence.append('image')
                            if self.image_dir[split] is not None:
                                image_path = os.path.join(self.image_dir[split], images[image_counter])
                                assert os.path.exists(image_path), f"Image path does not exist: {image_path}"
                                images[image_counter] = image_path
                            image_counter += 1
                    if i < len(context):
                        image_text_sequence.append('text')
                        texts.append(context[i])
                image_text_sequence.append('text')
                texts.append(f"\n\n{self.task_prompt}\nQuestion: {question}")                  
                    
                converted_samples[split].append({
                    "image_text_sequence": image_text_sequence,
                    "images": images,
                    "texts": texts,
                    "question": question,
                    "gt": gt,
                    "id": item["id"],
                    "image_to_find": image_to_find
                })
                
        return converted_samples