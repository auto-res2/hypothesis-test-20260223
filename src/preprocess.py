"""Dataset loading and preprocessing for GSM8K and ARC-Challenge."""

from datasets import load_dataset
from typing import Dict, List, Any
import re


def load_gsm8k(split: str = "test", cache_dir: str = ".cache") -> List[Dict[str, Any]]:
    """Load GSM8K dataset.
    
    Args:
        split: Dataset split (train/test)
        cache_dir: Directory to cache downloaded datasets
        
    Returns:
        List of examples with 'question', 'answer', and 'gold_answer' keys
    """
    dataset = load_dataset("gsm8k", "main", split=split, cache_dir=cache_dir)
    
    examples = []
    for item in dataset:
        # Extract numeric answer from the format "#### 123"
        answer_text = item["answer"]
        match = re.search(r"####\s*(-?\d+(?:\.\d+)?)", answer_text)
        gold_answer = match.group(1) if match else None
        
        examples.append({
            "question": item["question"],
            "answer": answer_text,  # Full answer with reasoning
            "gold_answer": gold_answer  # Numeric answer only
        })
    
    return examples


def load_arc_challenge(split: str = "test", cache_dir: str = ".cache") -> List[Dict[str, Any]]:
    """Load ARC-Challenge dataset.
    
    Args:
        split: Dataset split (train/validation/test)
        cache_dir: Directory to cache downloaded datasets
        
    Returns:
        List of examples with 'question', 'choices', 'answer_key', and 'gold_answer' keys
    """
    dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split, cache_dir=cache_dir)
    
    examples = []
    for item in dataset:
        choices = item["choices"]
        choice_text = "\n".join([
            f"{label}: {text}" 
            for label, text in zip(choices["label"], choices["text"])
        ])
        
        examples.append({
            "question": item["question"],
            "choices": choice_text,
            "choice_labels": choices["label"],
            "choice_texts": choices["text"],
            "answer_key": item["answerKey"],
            "gold_answer": item["answerKey"]  # The correct answer label (A/B/C/D)
        })
    
    return examples


def get_dataset(name: str, split: str = "test", cache_dir: str = ".cache") -> List[Dict[str, Any]]:
    """Load a dataset by name.
    
    Args:
        name: Dataset name (gsm8k or arc-challenge)
        split: Dataset split
        cache_dir: Directory to cache downloaded datasets
        
    Returns:
        List of dataset examples
    """
    if name == "gsm8k":
        return load_gsm8k(split=split, cache_dir=cache_dir)
    elif name == "arc-challenge":
        return load_arc_challenge(split=split, cache_dir=cache_dir)
    else:
        raise ValueError(f"Unknown dataset: {name}")
