# I-BAYGEN

## Install

```bash
pip install -r requirements.txt
```

## Prepare Model

Use any local model supported by `transformers.AutoModelForCausalLM`.

Example:

```bash
pip install modelscope
modelscope download --model iic/gte-Qwen2-1.5B-instruct --local_dir path/to/model
```

## Run

Input JSONL should contain a problem, candidate reasoning paths, and the ground truth code.

```bash
python rejection_sampling.py run \
  --data_path path/to/data.jsonl \
  --output_path outputs/selection.jsonl \
  --model_path path/to/model \
  --weight_mode linear
```

For a quick pipeline check without loading a model:

```bash
python rejection_sampling.py run \
  --data_path path/to/data.jsonl \
  --output_path outputs/output.jsonl \
  --limit 2
```

## Output

Each output row includes:

- `unweighted_selected`
- `weighted_selected`
- `all_scores`

`unweighted_selected` uses the sum of gold-code token log-probabilities.  
`weighted_selected` reweights gold-code token log-probabilities by token difficulty.
