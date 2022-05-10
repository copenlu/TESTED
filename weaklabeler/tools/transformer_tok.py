from transformers import AutoTokenizer
from typing import Dict, List

def transformer_tok(texts: List, tokenizer:AutoTokenizer) -> Dict: 
    return tokenizer(texts, padding = "max_length", truncation = True, return_tensors="pt")

