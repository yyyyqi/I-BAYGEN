# I-BAYGEN

This repository is a cleaned, runnable extraction of the I-BAYGEN rejection-sampling code used in the paper.

The core idea is simple:

1. For each LeetCode problem, sample multiple reasoning paths.
2. Append the gold solution after each reasoning path.
3. Score how likely the model finds the gold solution conditioned on that reasoning.
4. Select the reasoning path with the highest reward.

The baseline uses an unweighted log-prob reward. The paper method uses a weighted reward that gives more importance to harder ground-truth tokens.

## Repository Layout

```text
rejection_sampling.py     # generation, scoring, and comparison CLI
requirements.txt          # minimal runtime dependencies
outputs/                  # ignored local experiment files
```

The original project files that this extraction comes from are:

```text
LaTRO-main/LaTRO/utils/trainer_utils.py       # get_logprob_reward / get_weight_logprob_reward
LaTRO-main/LaTRO/trainer/latro_trainer.py     # weighted vs unweighted training loop
LeetCode/reject_sampling_data/*.jsonl         # LeetCode candidates and selected outputs
LeetCode/select_example.py                    # old comparison script
```

## Install

Use a Python environment with PyTorch and Transformers. For GPU inference, install a CUDA-compatible PyTorch build.

```bash
pip install -r requirements.txt
```

## Download A Model

Use any Hugging Face / ModelScope model that can be loaded by `transformers.AutoModelForCausalLM`.

Example with ModelScope:

```bash
pip install modelscope
modelscope download --model iic/gte-Qwen2-1.5B-instruct --local_dir path/to/models/gte-Qwen2-1.5B-instruct
```

Example with Hugging Face:

```bash
huggingface-cli download Alibaba-NLP/gte-Qwen2-1.5B-instruct --local-dir path/to/models/gte-Qwen2-1.5B-instruct
```

The GTE-Qwen2 model above can be loaded with standard Transformers as a causal language model for scoring existing reasoning paths. In our smoke test it was not reliable for free-form chain-of-thought generation, so use an instruction/chat code model if you need to generate fresh reasoning paths.

## Quick Smoke Test Without Loading a Model

This verifies the data path and output format using the existing candidate reasoning paths.

```bash
python rejection_sampling.py run \
  --data_path ../LeetCode/reject_sampling_data/reject_sampling_leetcode.jsonl \
  --output_path outputs/smoke_rejection_sampling.jsonl \
  --limit 2
```

The smoke-test scores are heuristic only. They are meant to test the pipeline quickly on a CPU machine.

## Run Paper-Style Scoring With Qwen

Set the model path to your downloaded local model:

```bash
MODEL_PATH="path/to/models/gte-Qwen2-1.5B-instruct"
```

Reuse existing candidate reasoning paths and select with both rewards:

```bash
python rejection_sampling.py run \
  --data_path ../LeetCode/reject_sampling_data/reject_sampling_leetcode.jsonl \
  --output_path outputs/qwen_reward_selection.jsonl \
  --model_path "$MODEL_PATH" \
  --limit 30 \
  --weight_mode linear
```

Generate fresh reasoning paths with a generative instruction model, then select:

```bash
python rejection_sampling.py run \
  --data_path ../LeetCode/reject_sampling_data/reject_sampling_leetcode.jsonl \
  --output_path outputs/qwen_generated_selection.jsonl \
  --model_path "$MODEL_PATH" \
  --force_generate \
  --num_candidates 4 \
  --temperature 0.7 \
  --max_new_tokens 512 \
  --limit 2 \
  --weight_mode linear
```

On a CPU-only machine, language-model scoring can be slow. A CUDA-enabled GPU is recommended.

## Compare Existing Weighted vs Unweighted Selections

If you have selected result files for both reward types, this command extracts cases where the selected reasoning differs:

```bash
python rejection_sampling.py compare-files \
  --unweighted_path ../LeetCode/reject_sampling_data/reject_sampling_train_avg_20.jsonl \
  --weighted_path ../LeetCode/reject_sampling_data/reject_sampling_train_weight_20.jsonl \
  --output_path outputs/existing_weighted_vs_unweighted.jsonl \
  --markdown_path outputs/existing_weighted_vs_unweighted.md \
  --only_different \
  --limit 10
```

Each output row contains:

- `problem`
- `unweighted_reasoning`
- `weighted_reasoning`
- `unweighted_code`
- `weighted_code`

This is the fastest way to inspect the qualitative effect of weighted reward.

## Reward Definition

For a problem `x`, candidate reasoning `r`, and gold code `y`, the unweighted reward is:

```text
R(r) = sum_t log p(y_t | x, r, y_<t)
```

The weighted reward computes token difficulty from the model's normalized log-prob over the gold-code tokens, then uses:

```text
R_weighted(r) = sum_t w_t log p(y_t | x, r, y_<t) / sum_t w_t
```

Supported weighting modes:

- `linear`: `w_t = 1 - normalized_logprob_t`
- `inverse`: `w_t = 1 / normalized_logprob_t`, clipped
- `power`: `w_t = (1 - normalized_logprob_t)^2`

## Notes For GitHub Release

Before pushing, keep large files out of Git:

```gitignore
outputs/
*.safetensors
checkpoint-*/
results*/
__pycache__/
```

Do not commit the local model directory. Tell users to download or point to their own model path.
