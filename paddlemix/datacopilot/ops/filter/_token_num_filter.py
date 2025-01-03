import os
from typing import Optional
from functools import partial
from ...core import MMDataset, register
from paddlenlp.transformers import AutoTokenizer
import sys


# Define the function to compute token count
def compute_token_count(user_conv: str, tokenizer: AutoTokenizer) -> int:
    """
    Compute the number of tokens in the sample (conversation).

    Args:
        user_conv (str): Merged conversation text.
        tokenizer (AutoTokenizer): Tokenizer instance used for tokenization.

    Returns:
        int: The number of tokens in the sample.
    """
    tokens = tokenizer(user_conv, truncation=True, return_tensors="pd", use_fast=True)["input_ids"].flatten()
    return len(tokens)

@register()
def token_num_filter(
    dataset, 
    tokenizer_model: str = "Qwen/Qwen2.5-7B", 
    min_tokens: Optional[int] = 10, 
    max_tokens: Optional[int] = sys.maxsize
) -> MMDataset:
    """
    Filter the dataset based on the number of tokens in each sample.

    Args:
        dataset (MMDataset): The dataset to be filtered.
        tokenizer_model (str): Name of the tokenizer model to use, default is `Qwen/Qwen2.5-7B`.
        min_tokens (int): Minimum number of tokens. Default is 10.
        max_tokens (int): Maximum number of tokens. Default is `sys.maxsize`.

    Returns:
        MMDataset: The filtered dataset.
    """
    print(f"Filtering samples based on token count: min tokens = {min_tokens}, max tokens = {max_tokens}...")
    
    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)

    def filter_func(item):
        # Get and clean conversation text
        user_conv = '\n\n'.join(
            ''.join(conversation) for conversation in item['conversations']
        ).replace('<image>', '').replace('\n', '')  # Clean `<image>` tags and newlines

        # Compute the number of tokens
        num_tokens = compute_token_count(user_conv, tokenizer)

        # Check if the token count is within the specified range
        return min_tokens <= num_tokens <= max_tokens

    # Apply dataset.filter
    filtered_dataset = dataset.filter(
        func=filter_func, 
        max_workers=8, 
        progress=True
    )
    
    return filtered_dataset