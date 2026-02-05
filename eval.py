import argparse
import datetime
import os
from tqdm import tqdm
import json
import re
import string
import unicodedata
import numpy as np


# =========================
# Normalization / CQA Eval
# =========================

def normalize_answer(s: str) -> str:
    """Lowercase, remove punctuation/articles/extra whitespace."""
    s = unicodedata.normalize("NFD", s)

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s: str):
    return normalize_answer(s).split()


def compute_exact(gold: str, pred: str) -> int:
    return int(normalize_answer(gold) == normalize_answer(pred))


def compute_prf1(gold: str, pred: str):
    """Return precision, recall, f1 for one pair (SQuAD-style token overlap)."""
    gold_toks = get_tokens(gold)
    pred_toks = get_tokens(pred)
    common = set(gold_toks) & set(pred_toks)
    num_same = sum(min(gold_toks.count(tok), pred_toks.count(tok)) for tok in common)

    if num_same == 0:
        return 0.0, 0.0, 0.0

    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def evaluate(golds, predictions):
    ems, precisions, recalls, f1s = [], [], [], []

    for gold, pred in zip(golds, predictions):
        ems.append(compute_exact(gold, pred))
        p, r, f1 = compute_prf1(gold, pred)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

    em = np.mean(ems)
    p = np.mean(precisions)
    r = np.mean(recalls)
    f1 = np.mean(f1s)

    return {
        "EM": em,
        "F1": f1
    }, precisions, recalls, f1s


# =========================
# VLM Inference
# =========================

class VLMInference:
    def __init__(self, model, model_name_of_path: str,
                 use_vllm=False, sampling_params=None,
                 generate_kwargs={}) -> None:

        self.sampling_params = sampling_params
        self.model = None

        if 'phi' in model_name_of_path.lower():
            from vlm.phi import PhiVision
            self.model = PhiVision(model_path=model_name_of_path)

        if 'internvl' in model_name_of_path.lower():
            from vlm.internvl import InternVL
            self.model = InternVL(model_path=model_name_of_path)

        if 'idefics' in model_name_of_path.lower():
            from vlm.idefics2 import Idefics2
            self.model = Idefics2(
                model_path=model_name_of_path,
                interleaved_visuals=True
            )

        if 'kimi' in model_name_of_path.lower():
            from vlm.kimi_vl import KimiVL
            if use_vllm:
                self.model = KimiVL(
                    model=model,
                    model_path=model_name_of_path,
                    use_vllm=use_vllm,
                    interleaved_visuals=True
                )
            else:
                self.model = KimiVL(
                    model_path=model_name_of_path,
                    use_vllm=use_vllm,
                    interleaved_visuals=True
                )

        if 'llama' in model_name_of_path.lower():
            from vlm.llama_vision import LlamaVision
            if use_vllm:
                self.model = LlamaVision(
                    model=model,
                    model_path=model_name_of_path,
                    use_vllm=use_vllm,
                    interleaved_visuals=True
                )
            else:
                self.model = LlamaVision(
                    model_path=model_name_of_path,
                    use_vllm=use_vllm,
                    interleaved_visuals=True
                )

        if 'qwen2.5-vl' in model_name_of_path.lower():
            from vlm.qwen2_5_vl import Qwen2_5_VL
            if use_vllm:
                self.model = Qwen2_5_VL(
                    model=model,
                    model_path=model_name_of_path,
                    use_vllm=use_vllm,
                    interleaved_visuals=True
                )
            else:
                self.model = Qwen2_5_VL(
                    model_path=model_name_of_path,
                    use_vllm=use_vllm,
                    interleaved_visuals=True
                )

        if 'qwen3-vl' in model_name_of_path.lower():
            from vlm.qwen3_vl import Qwen3_VL
            if use_vllm:
                self.model = Qwen3_VL(model=model, model_path=model_name_of_path, use_vllm=use_vllm, interleaved_visuals=True)
            else:
                self.model = Qwen3_VL(model_path=model_name_of_path, use_vllm=use_vllm, interleaved_visuals=True)
                
        if 'llava-onevision' in model_name_of_path.lower():
            from vlm.llava_onevision import LLaVAOneVision
            if use_vllm:
                self.model = LLaVAOneVision(
                    model=model,
                    model_path=model_name_of_path,
                    use_vllm=use_vllm,
                    interleaved_visuals=True
                )
            else:
                self.model = LLaVAOneVision(
                    model_path=model_name_of_path,
                    use_vllm=use_vllm,
                    interleaved_visuals=True
                )

        self.generate_kwargs = generate_kwargs

    def generate(self, prepared_input):

        if self.sampling_params is not None:
            output = self.model(
                prepared_input,
                sampling_params=self.sampling_params,
                generate_kwargs=self.generate_kwargs
            )
        else:
            output = self.model(
                prepared_input,
                generate_kwargs=self.generate_kwargs
            )

        return output


# =========================
# Benchmark Evaluator (CRIT only)
# =========================

class BenchmarkEvaluator:
    def __init__(self,
                 benchmark_name: str,
                 model_name_of_path: str,
                 use_cot: bool,
                 vlm: VLMInference,
                 use_crit_model: bool = False) -> None:

        self.benchmark_name = benchmark_name
        self.vlm = vlm
        self.use_cot = use_cot

        from crit import CRIT
        self.dataset = CRIT(
            use_cot=self.use_cot,
        )

        self.output_path = (
            f"./outputs/CRIT/"
            f"{model_name_of_path.split('/')[-1]}/"
            f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        )
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

    # -------- Input prep --------

    def prepare_input(self, image_text_sequence, images, texts):

        prepared_input = []
        image_counter, text_counter = 0, 0

        for i, item in enumerate(image_text_sequence):

            if item == 'image':
                prepared_input.append({
                    'type': 'image',
                    'content': images[image_counter]
                })
                image_counter += 1

            elif item == 'text':

                if i > 0 and image_text_sequence[i - 1] == 'text':
                    prepared_input[-1]['content'] += "\n" + texts[text_counter]
                else:
                    prepared_input.append({
                        'type': 'text',
                        'content': texts[text_counter]
                    })

                text_counter += 1

        return prepared_input

    # -------- Evaluation --------

    def evaluate(self):

        samples = self.dataset.get_samples_by_split()
        results = []

        print(
            f"Evaluating "
            f"{len(samples['natural_image']) + len(samples['video']) + len(samples['scientific_paper'])} "
            f"samples for CRIT"
        )

        for split, data in samples.items():
            for item in tqdm(data, total=len(data)):

                image_text_sequence = item["image_text_sequence"]
                images = item["images"]
                texts = item["texts"]
                gt = item["gt"]
                question = item["question"]

                prepared_input = self.prepare_input(
                    image_text_sequence,
                    images,
                    texts
                )

                prediction = self.vlm.generate(prepared_input)

                if (
                    isinstance(prediction, dict)
                    and 'thinking' in prediction
                    and 'summary' in prediction
                ):
                    final_answer = prediction['summary']
                    thinking = prediction['thinking']
                else:
                    final_answer = prediction
                    thinking = ""

                results.append({
                    "split": split,
                    "question": question,
                    "prediction": final_answer,
                    "gt": gt
                })

                result = {
                    "split": split,
                    "id": item['id'],
                    "question": question,
                    "original_prediction": prediction,
                    "thinking": thinking,
                    "parsed_prediction":
                        prediction.split('Final Answer:')[-1].strip()
                        if isinstance(prediction, str)
                        else final_answer,
                    "gt": gt
                }

                with open(self.output_path, 'a') as f:
                    f.write(json.dumps(result) + "\n")

        return results

    # -------- Scoring --------

    def calculate_score(self, results):

        for split in ["natural_image"]:
            split_results = [
                r for r in results
                if r["split"] == split
            ]

            golds = [r["gt"] for r in split_results]
            predictions = [
                r["prediction"].split("Answer:")[-1].strip()
                for r in split_results
            ]

            scores, _, _, _ = evaluate(golds, predictions)
            print(f"Scores on {split} split: {scores}")

        golds = [r["gt"] for r in results]
        predictions = [
            r["prediction"].split("Answer:")[-1].strip()
            for r in results
        ]

        scores, _, _, _ = evaluate(golds, predictions)
        print(f"Overall Scores: {scores}")

        return scores["F1"]


# =========================
# Main
# =========================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_of_path", type=str, required=True)
    parser.add_argument("--use_cot", action="store_true")
    parser.add_argument("--use_vllm", action="store_true")

    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=256)

    args = parser.parse_args()

    generate_kwargs = {
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens
    }

    vlm = VLMInference(
        model=None,
        model_name_of_path=args.model_name_of_path,
        use_vllm=args.use_vllm,
        generate_kwargs=generate_kwargs
    )

    evaluator = BenchmarkEvaluator(
        benchmark_name="CRIT",
        model_name_of_path=args.model_name_of_path,
        use_cot=args.use_cot,
        vlm=vlm,
        use_crit_model=args.use_crit_model
    )

    results = evaluator.evaluate()
    score = evaluator.calculate_score(results)

    print(f"Final score on CRIT: {score}")
