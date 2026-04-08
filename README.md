# CRIT Evaluation

This repository provides a single evaluation entry point for the CRIT benchmark: `eval.py`. This README focuses only on running evaluation, not on the internal implementation.

## What You Need

Place the benchmark annotations and image assets in the repository root using the paths expected by `crit.py`.

Required annotation files:

- `natural_image_benchmark_total_for_eval_refined_wo_cross_image.json`
- `video_benchmark_total_for_eval_refined.json`
- `scientific_paper_benchmark_total_for_eval_refined.json`

Required asset directories:

- `data/`
- `data/ActivityNet-Captions/`
- `data/spiqa/train_val/`

The evaluator will assert that every referenced image path exists.

## Main Evaluation Command

The main interface is:

```bash
python eval.py \
  --model_name_of_path <model_name_or_path> \
  --temperature 0 \
  --max_new_tokens 1024 \
  --use_cot
```

Example:

```bash
python eval.py \
  --model_name_of_path Qwen/Qwen2.5-VL-7B-Instruct \
  --temperature 0 \
  --max_new_tokens 1024 \
  --use_cot
```

There is also a convenience script:

```bash
bash eval_crit.sh
```

## Arguments

- `--model_name_of_path`: Hugging Face model name or local model path. Model dispatch is selected from this string.
- `--use_cot`: Uses the benchmark prompt that asks the model to reason step by step and end with `Final Answer: <answer>`.
- `--use_vllm`: Enables the vLLM code path for wrappers that support it.
- `--temperature`: Sampling temperature passed to generation.
- `--max_new_tokens`: Maximum number of generated tokens.

## Output

Per-example predictions are written to:

```text
outputs/CRIT/<model_name>/<timestamp>.jsonl
```

The script prints:

- natural image split scores
- overall CRIT scores
- final F1 score

Scoring uses exact match and token-overlap F1 after answer normalization.

## Supported VLMs in `eval.py`

`eval.py` currently routes to the following wrappers based on `--model_name_of_path`:

- Qwen2.5-VL: use a name containing `qwen2.5-vl`
- Qwen3-VL: use a name containing `qwen3-vl`
- Kimi-VL: use a name containing `kimi`
- Phi Vision: use a name containing `phi`
- InternVL: use a name containing `internvl`
- Idefics2: use a name containing `idefics`
- Llama Vision: use a name containing `llama`
- LLaVA-OneVision: use a name containing `llava-onevision`

Examples:

```bash
python eval.py --model_name_of_path Qwen/Qwen2.5-VL-7B-Instruct --temperature 0 --max_new_tokens 1024 --use_cot
python eval.py --model_name_of_path Qwen/Qwen3-VL-8B-Instruct --temperature 0 --max_new_tokens 1024 --use_cot
python eval.py --model_name_of_path moonshotai/Kimi-VL-A3B-Thinking-2506 --temperature 0 --max_new_tokens 1024 --use_cot
python eval.py --model_name_of_path microsoft/Phi-3.5-vision-instruct --temperature 0 --max_new_tokens 1024 --use_cot
python eval.py --model_name_of_path OpenGVLab/InternVL2_5-8B --temperature 0 --max_new_tokens 1024 --use_cot
python eval.py --model_name_of_path HuggingFaceM4/idefics2-8b --temperature 0 --max_new_tokens 1024 --use_cot
python eval.py --model_name_of_path meta-llama/Llama-3.2-11B-Vision --temperature 0 --max_new_tokens 1024 --use_cot
python eval.py --model_name_of_path llava-hf/llava-onevision-qwen2-7b-ov-hf --temperature 0 --max_new_tokens 1024 --use_cot
```

## Model-Specific Notes

### Qwen2.5-VL

Works through the default Hugging Face path. Add `--use_vllm` only if you have separately wired a valid vLLM model instance into `VLMInference`; `eval.py` does not currently instantiate one for you.

### Qwen3-VL

Same usage pattern as Qwen2.5-VL. The wrapper supports standard Hugging Face loading. The `--use_vllm` flag has the same limitation as above.

### Kimi-VL

The default non-vLLM path uses the Hugging Face/Transformers wrapper and supports the model’s thinking-style output. The evaluator stores both the thinking trace and the summarized final answer when present.

### Phi-3.5 Vision

Use the normal command without `--use_vllm`.

### InternVL

Use the normal command without `--use_vllm`. The wrapper expects CUDA and uses the model’s chat interface directly.

### Idefics2

Use the normal command without `--use_vllm`.

### Llama Vision

Do not pass `--use_vllm`. The current wrapper raises `NotImplementedError` for the vLLM path.

### LLaVA-OneVision

Do not pass `--use_vllm`. The current wrapper raises `NotImplementedError` for the vLLM path.

## Evaluating Other VLMs

If your model is not one of the routed names above, there are two practical options.

### Option 1: Reuse an Existing Wrapper

If your model is API-compatible with one of the existing Hugging Face wrappers, pass a `--model_name_of_path` string that matches that wrapper’s dispatch rule and points to your checkpoint or local path.

### Option 2: Add a Model-Specific Wrapper

If your model needs its own preprocessing, prompt template, or API call pattern:

1. Add a wrapper under `vlm/` with a `__call__(inputs, generate_kwargs=...)` interface.
2. Add a dispatch branch in `VLMInference` inside `eval.py`.
3. Run the same `python eval.py ...` command with your new model identifier.

This is the correct path for models that require provider-specific SDKs or custom multimodal formatting.

## OpenAI and Gemini

The repository already includes:

- `vlm/openai.py`
- `vlm/gemini.py`

However, these wrappers are not currently connected to `eval.py`, so you cannot evaluate them with the stock command yet.

To use them, add a dispatch branch in `VLMInference` and then run the same evaluator. In practice:

1. Choose a model identifier such as `gpt-5`, `gpt-4o`, or `gemini-2.5-flash-lite`.
2. Add a corresponding condition in `eval.py` that instantiates `OpenAIVision` or `Gemini`.
3. Set the provider credentials required by that SDK before running evaluation.

After that, evaluation still happens through:

```bash
python eval.py --model_name_of_path <your_api_model_name> --temperature 0 --max_new_tokens 1024 --use_cot
```

## Recommended Command Pattern

For most local Hugging Face VLMs:

```bash
python eval.py \
  --model_name_of_path <hf_model_or_local_checkpoint> \
  --temperature 0 \
  --max_new_tokens 1024 \
  --use_cot
```

Avoid `--use_vllm` unless you have explicitly implemented and tested the vLLM path for that model in this repository.
