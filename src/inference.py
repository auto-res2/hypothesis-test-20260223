"""Inference script for running LLM evaluations."""

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from omegaconf import DictConfig, OmegaConf
import wandb
from openai import AsyncOpenAI
from tqdm import tqdm
import sys

from src.preprocess import get_dataset


def parse_numeric_answer(text: str) -> Optional[str]:
    """Extract numeric answer from model output.
    
    Looks for patterns like "The answer is: 123" or "#### 123"
    """
    # Look for "The answer is: NUMBER"
    match = re.search(r"[Tt]he answer is:?\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)", text)
    if match:
        # Remove commas from number
        return match.group(1).replace(",", "")
    
    # Look for "#### NUMBER" pattern
    match = re.search(r"####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(",", "")
    
    # Look for any number at the end
    match = re.search(r"(-?\d+(?:,\d{3})*(?:\.\d+)?)\s*$", text.strip())
    if match:
        return match.group(1).replace(",", "")
    
    return None


def parse_multiple_choice_answer(text: str) -> Optional[str]:
    """Extract multiple choice answer (A/B/C/D) from model output."""
    # Look for "The answer is: A"
    match = re.search(r"[Tt]he answer is:?\s*([A-D])", text)
    if match:
        return match.group(1)
    
    # Look for standalone letter at the end
    match = re.search(r"\b([A-D])\b\s*$", text.strip())
    if match:
        return match.group(1)
    
    # Look for any occurrence of A/B/C/D near the end
    matches = re.findall(r"\b([A-D])\b", text[-100:])
    if matches:
        return matches[-1]
    
    return None


async def query_openai_async(
    client: AsyncOpenAI,
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    semaphore: asyncio.Semaphore
) -> Dict[str, Any]:
    """Query OpenAI API asynchronously with rate limiting."""
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return {
                "response": response.choices[0].message.content,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
        except Exception as e:
            return {
                "response": None,
                "error": str(e)
            }


async def run_inference_async(cfg: DictConfig):
    """Run inference on dataset using LLM API."""
    # Load dataset
    print(f"Loading dataset: {cfg.run.dataset.name}")
    dataset = get_dataset(
        name=cfg.run.dataset.name,
        split=cfg.run.dataset.split,
        cache_dir=cfg.run.dataset.cache_dir
    )
    
    # Limit dataset size in sanity_check mode
    if cfg.mode == "sanity_check":
        dataset = dataset[:10]
        print(f"Sanity check mode: using {len(dataset)} samples")
    
    print(f"Dataset loaded: {len(dataset)} examples")
    
    # Initialize OpenAI client
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Create semaphore for rate limiting
    max_concurrent = cfg.run.inference.max_concurrent
    if cfg.mode == "sanity_check":
        max_concurrent = min(max_concurrent, 8)
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Create prompts
    prompt_template = cfg.run.method.prompt_template
    prompts = []
    for example in dataset:
        if cfg.run.dataset.name == "gsm8k":
            prompt = prompt_template.format(question=example["question"])
        elif cfg.run.dataset.name == "arc-challenge":
            prompt = prompt_template.format(
                question=example["question"],
                choices=example["choices"]
            )
        else:
            raise ValueError(f"Unknown dataset: {cfg.run.dataset.name}")
        prompts.append(prompt)
    
    # [VALIDATOR FIX - Attempt 1]
    # [PROBLEM]: 0% accuracy - model responses are misaligned with questions
    # [CAUSE]: asyncio.as_completed() returns futures in completion order, not submission order,
    #          causing responses to be paired with wrong examples
    # [FIX]: Use asyncio.gather() to preserve order of results
    #
    # [OLD CODE]:
    # results = []
    # for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Querying API"):
    #     result = await f
    #     results.append(result)
    #
    # [NEW CODE]:
    # Run inference
    print(f"Running inference with {max_concurrent} concurrent requests...")
    tasks = [
        query_openai_async(
            client=client,
            prompt=prompt,
            model=cfg.run.model.name,
            temperature=cfg.run.model.temperature,
            max_tokens=cfg.run.model.max_tokens,
            semaphore=semaphore
        )
        for prompt in prompts
    ]
    
    # Use asyncio.gather to preserve order
    results = []
    with tqdm(total=len(tasks), desc="Querying API") as pbar:
        # Gather all results while maintaining order
        results = await asyncio.gather(*tasks)
        pbar.update(len(tasks))
    
    # Parse answers and compute accuracy
    correct = 0
    total = 0
    unparseable = 0
    
    answer_format = cfg.run.inference.answer_format
    parsed_results = []
    
    for i, (example, result) in enumerate(zip(dataset, results)):
        response_text = result.get("response", "")
        gold_answer = example["gold_answer"]
        
        if response_text is None:
            parsed_answer = None
            unparseable += 1
        elif answer_format == "numeric":
            parsed_answer = parse_numeric_answer(response_text)
            if parsed_answer is None:
                unparseable += 1
        elif answer_format == "multiple_choice":
            parsed_answer = parse_multiple_choice_answer(response_text)
            if parsed_answer is None:
                unparseable += 1
        else:
            raise ValueError(f"Unknown answer format: {answer_format}")
        
        is_correct = False
        if parsed_answer is not None and gold_answer is not None:
            # Normalize answers for comparison
            if answer_format == "numeric":
                try:
                    is_correct = float(parsed_answer) == float(gold_answer)
                except (ValueError, TypeError):
                    is_correct = parsed_answer == gold_answer
            else:
                is_correct = parsed_answer == gold_answer
            
            if is_correct:
                correct += 1
            total += 1
        
        parsed_results.append({
            "example_id": i,
            "question": example["question"],
            "gold_answer": gold_answer,
            "model_response": response_text,
            "parsed_answer": parsed_answer,
            "is_correct": is_correct,
            "usage": result.get("usage", {}),
            "error": result.get("error", None)
        })
    
    accuracy = correct / total if total > 0 else 0.0
    
    # Log to WandB
    if cfg.wandb.mode == "online":
        wandb.log({
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "unparseable": unparseable,
            "samples_processed": len(dataset)
        })
        wandb.summary["final_accuracy"] = accuracy
        wandb.summary["correct"] = correct
        wandb.summary["total"] = total
        wandb.summary["unparseable"] = unparseable
    
    # Save results
    results_dir = Path(cfg.results_dir) / cfg.run.run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / "results.json", "w") as f:
        json.dump(parsed_results, f, indent=2)
    
    with open(results_dir / "metrics.json", "w") as f:
        json.dump({
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "unparseable": unparseable,
            "samples_processed": len(dataset)
        }, f, indent=2)
    
    print(f"\nResults saved to {results_dir}")
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"Unparseable: {unparseable}")
    
    # Sanity validation
    if cfg.mode == "sanity_check":
        perform_sanity_validation(parsed_results, accuracy, total)
    
    return accuracy


def perform_sanity_validation(results: List[Dict], accuracy: float, total: int):
    """Perform sanity validation checks for inference task."""
    samples_processed = len(results)
    outputs_valid = sum(1 for r in results if r["model_response"] is not None)
    outputs_unique = len(set(r["model_response"] for r in results if r["model_response"] is not None))
    
    # Print summary
    summary = {
        "samples": samples_processed,
        "outputs_valid": outputs_valid,
        "outputs_unique": outputs_unique,
        "accuracy": accuracy,
        "total": total
    }
    print(f"SANITY_VALIDATION_SUMMARY: {json.dumps(summary)}")
    
    # Check validation conditions
    if samples_processed < 5:
        print("SANITY_VALIDATION: FAIL reason=insufficient_samples")
        sys.exit(1)
    
    if outputs_valid < 5:
        print("SANITY_VALIDATION: FAIL reason=insufficient_valid_outputs")
        sys.exit(1)
    
    if outputs_unique < 2:
        print("SANITY_VALIDATION: FAIL reason=all_outputs_identical")
        sys.exit(1)
    
    if total == 0:
        print("SANITY_VALIDATION: FAIL reason=no_parseable_answers")
        sys.exit(1)
    
    print("SANITY_VALIDATION: PASS")


def run_inference(cfg: DictConfig):
    """Main entry point for inference."""
    # Initialize WandB
    if cfg.wandb.mode == "online":
        # Override project for sanity check
        project = cfg.wandb.project
        if cfg.mode == "sanity_check":
            project = f"{project}-sanity"
        
        run = wandb.init(
            entity=cfg.wandb.entity,
            project=project,
            name=cfg.run.run_id,
            config=OmegaConf.to_container(cfg, resolve=True)
        )
        print(f"WandB run URL: {run.url}")
    
    # Run inference
    accuracy = asyncio.run(run_inference_async(cfg))
    
    if cfg.wandb.mode == "online":
        wandb.finish()
    
    return accuracy


if __name__ == "__main__":
    # This script is invoked by main.py
    # Config is passed via hydra
    pass
