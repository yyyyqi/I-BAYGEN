import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_reasoning(text: str) -> str:
    """Return the reasoning part before the first Python code fence."""
    if not text:
        return ""
    return text.split("```python", 1)[0].strip()


def extract_python_code(text: str) -> str:
    if not text:
        return ""
    match = re.search(r"```python\s*(.*?)```", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def qwen_chat_prompt(user_prompt: str) -> str:
    return (
        "<|im_start|>system\n"
        "You are Qwen, created by Alibaba Cloud. You are a helpful programming assistant.\n"
        "<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt.strip()}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def build_reasoning_prompt(problem: str) -> str:
    return (
        "Solve the following LeetCode problem. First write a concise step-by-step "
        "reasoning process, then provide the final Python code in a ```python code block.\n\n"
        f"{problem.strip()}"
    )


def build_reward_text(problem: str, candidate_reasoning: str, gold_code: str) -> str:
    return (
        qwen_chat_prompt(build_reasoning_prompt(problem))
        + candidate_reasoning.strip()
        + "\n\n```python\n"
        + extract_python_code(gold_code)
        + "\n```"
    )


def load_model(model_path: str, trust_remote_code: bool = False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = "auto"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    return model, tokenizer


def generate_candidates(model, tokenizer, problem: str, num_candidates: int, max_new_tokens: int, temperature: float) -> List[str]:
    import torch

    prompt = qwen_chat_prompt(build_reasoning_prompt(problem))
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    do_sample = num_candidates > 1 or temperature > 0
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=max(temperature, 1e-5),
            top_p=0.95,
            num_return_sequences=num_candidates,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    prompt_len = inputs["input_ids"].shape[1]
    return [
        tokenizer.decode(output[prompt_len:], skip_special_tokens=True).strip()
        for output in outputs
    ]


def score_candidate_pair(model, tokenizer, problem: str, candidate: str, gold_code: str, mode: str) -> Dict[str, float]:
    import torch

    reasoning = extract_reasoning(candidate)
    prefix = qwen_chat_prompt(build_reasoning_prompt(problem)) + reasoning.strip() + "\n\n```python\n"
    target = extract_python_code(gold_code) + "\n```"
    full_text = prefix + target

    full = tokenizer(full_text, return_tensors="pt").to(model.device)
    prefix_ids = tokenizer(prefix, return_tensors="pt")["input_ids"]
    target_start = prefix_ids.shape[1]

    with torch.no_grad():
        logits = model(**full).logits
        logprobs = torch.log_softmax(logits[:, :-1, :], dim=-1)
        next_ids = full["input_ids"][:, 1:]
        token_logprobs = torch.gather(logprobs, 2, next_ids.unsqueeze(-1)).squeeze(-1)[0]

    mask = torch.arange(token_logprobs.shape[0], device=token_logprobs.device) >= (target_start - 1)
    target_logprobs = token_logprobs[mask]
    if target_logprobs.numel() == 0:
        return {"unweighted_reward": -math.inf, "weighted_reward": -math.inf}

    unweighted_reward = target_logprobs.sum().item()

    eps = 1e-8
    min_lp = target_logprobs.min()
    max_lp = target_logprobs.max()
    normalized = (target_logprobs - min_lp) / (max_lp - min_lp + eps)
    if mode == "linear":
        weights = 1.0 - normalized
    elif mode == "inverse":
        weights = torch.clamp(1.0 / (normalized + eps), 0, 100.0)
    elif mode == "power":
        weights = (1.0 - normalized) ** 2
    else:
        raise ValueError(f"Unsupported weighting mode: {mode}")
    weighted_reward = ((target_logprobs * weights).sum() / (weights.sum() + eps)).item()
    return {"unweighted_reward": unweighted_reward, "weighted_reward": weighted_reward}


def select_from_existing(row: Dict[str, Any], mode: str = "mock") -> Dict[str, Any]:
    """Fast no-model selector for smoke tests; use --model_path for real rewards."""
    candidates = row.get("cots") or []
    if not candidates:
        candidates = [extract_reasoning(row.get("code", ""))]
    scored = []
    for i, candidate in enumerate(candidates):
        words = len(candidate.split())
        code_like = 1 if "class Solution" in candidate or "def " in candidate else 0
        unweighted = words + 25 * code_like
        weighted = words + 40 * code_like + 3 * candidate.lower().count("edge")
        scored.append({"candidate_id": i, "text": candidate, "unweighted_reward": unweighted, "weighted_reward": weighted})
    return {
        "index": row.get("index", row.get("task_id")),
        "problem": row.get("problem", row.get("prompt")),
        "unweighted_selected": max(scored, key=lambda x: x["unweighted_reward"]),
        "weighted_selected": max(scored, key=lambda x: x["weighted_reward"]),
        "all_scores": scored,
        "note": f"{mode} scores are for pipeline smoke tests only; use --model_path for paper-style log-prob rewards.",
    }


def run_rejection_sampling(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.data_path)
    if args.indices:
        wanted = {item.strip() for item in args.indices.split(",") if item.strip()}
        rows = [row for row in rows if str(row.get("index", row.get("task_id"))) in wanted]
    rows = rows[: args.limit]
    results = []

    if args.model_path:
        model, tokenizer = load_model(args.model_path, trust_remote_code=args.trust_remote_code)
        finished = []
        if args.resume and Path(args.output_path).exists():
            finished = read_jsonl(args.output_path)
        finished_ids = {str(row.get("index")) for row in finished}
        results = finished
        for row in rows:
            row_id = str(row.get("index", row.get("task_id")))
            if row_id in finished_ids:
                continue
            problem = row.get("problem", row.get("prompt", ""))
            gold_code = row.get("code", "")
            candidates = row.get("cots")
            if not candidates or args.force_generate:
                candidates = generate_candidates(model, tokenizer, problem, args.num_candidates, args.max_new_tokens, args.temperature)
            if args.candidate_limit:
                candidates = candidates[: args.candidate_limit]
            scored = []
            for i, candidate in enumerate(candidates):
                rewards = score_candidate_pair(model, tokenizer, problem, candidate, gold_code, args.weight_mode)
                scored.append(
                    {
                        "candidate_id": i,
                        "text": candidate,
                        **rewards,
                    }
                )
            result = {
                "index": row.get("index", row.get("task_id")),
                "problem": problem,
                "unweighted_selected": max(scored, key=lambda x: x["unweighted_reward"]),
                "weighted_selected": max(scored, key=lambda x: x["weighted_reward"]),
                "all_scores": scored,
            }
            results.append(result)
            write_jsonl(args.output_path, results)
    else:
        results = [select_from_existing(row) for row in rows]

    write_jsonl(args.output_path, results)


def compare_selected_files(args: argparse.Namespace) -> None:
    avg_rows = {row["index"]: row for row in read_jsonl(args.unweighted_path)}
    weighted_rows = {row["index"]: row for row in read_jsonl(args.weighted_path)}
    rows = []
    for index in sorted(set(avg_rows) & set(weighted_rows)):
        avg = avg_rows[index]
        weighted = weighted_rows[index]
        avg_reasoning = extract_reasoning(avg.get("code", ""))
        weighted_reasoning = extract_reasoning(weighted.get("code", ""))
        if args.only_different and avg_reasoning == weighted_reasoning:
            continue
        rows.append(
            {
                "index": index,
                "problem": avg.get("problem", weighted.get("problem", "")),
                "unweighted_reasoning": avg_reasoning,
                "weighted_reasoning": weighted_reasoning,
                "unweighted_code": avg.get("code", ""),
                "weighted_code": weighted.get("code", ""),
            }
        )
        if len(rows) >= args.limit:
            break
    write_jsonl(args.output_path, rows)
    if args.markdown_path:
        write_markdown_comparison(args.markdown_path, rows)


def clip_text(text: str, max_chars: int = 1800) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n\n...[truncated]"


def write_markdown_comparison(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    parts = ["# Weighted vs Unweighted Reasoning Comparison\n"]
    for row in rows:
        title = row.get("index", "unknown")
        problem = clip_text(row.get("problem", ""), 1200)
        unweighted = clip_text(row.get("unweighted_reasoning", ""))
        weighted = clip_text(row.get("weighted_reasoning", ""))
        parts.append(f"## Problem {title}\n")
        parts.append("### Problem\n")
        parts.append(f"```text\n{problem}\n```\n")
        parts.append("### Unweighted Reward Selection\n")
        parts.append(f"```text\n{unweighted}\n```\n")
        parts.append("### Weighted Reward Selection\n")
        parts.append(f"```text\n{weighted}\n```\n")
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rejection sampling demo for weighted reward vs unweighted reward.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Generate or reuse candidates, then select with both rewards.")
    run_parser.add_argument("--data_path", required=True)
    run_parser.add_argument("--output_path", required=True)
    run_parser.add_argument("--model_path", default=None)
    run_parser.add_argument("--trust_remote_code", action="store_true")
    run_parser.add_argument("--resume", action="store_true")
    run_parser.add_argument("--limit", type=int, default=2)
    run_parser.add_argument("--indices", default=None, help="Comma-separated index/task_id values to run.")
    run_parser.add_argument("--num_candidates", type=int, default=4)
    run_parser.add_argument("--candidate_limit", type=int, default=None)
    run_parser.add_argument("--max_new_tokens", type=int, default=512)
    run_parser.add_argument("--temperature", type=float, default=0.7)
    run_parser.add_argument("--weight_mode", choices=["linear", "inverse", "power"], default="linear")
    run_parser.add_argument("--force_generate", action="store_true")
    run_parser.set_defaults(func=run_rejection_sampling)

    cmp_parser = subparsers.add_parser("compare-files", help="Compare existing selected weighted/unweighted JSONL files.")
    cmp_parser.add_argument("--unweighted_path", required=True)
    cmp_parser.add_argument("--weighted_path", required=True)
    cmp_parser.add_argument("--output_path", required=True)
    cmp_parser.add_argument("--markdown_path", default=None)
    cmp_parser.add_argument("--limit", type=int, default=20)
    cmp_parser.add_argument("--only_different", action="store_true")
    cmp_parser.set_defaults(func=compare_selected_files)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
