#!/usr/bin/env python3
"""
sentiment_transformer_demo.py

A simple script that uses Hugging Face’s DistilBERT‐SST2 model to compute
POSITIVE/NEGATIVE probabilities on a few example sentences, demonstrating
that the transformer‐based sentiment analysis works.

Usage:
    pip install transformers torch
    python sentiment_transformer_demo.py
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. Choose the pretrained model
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

# 2. Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()  # disable dropout
device = torch.device("cpu")
model.to(device)

def analyze_sentence(text: str):
    """
    Tokenizes `text`, pads/truncates to the model’s max length,
    runs it through DistilBERT‐SST2, and returns a dict:
      {
        "neg_prob": <float>,
        "pos_prob": <float>,
        "score":    pos_prob - neg_prob
      }
    """
    # 1. Tokenize, adding special tokens. Return PyTorch tensors on CPU.
    encoded = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=tokenizer.model_max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids      = encoded["input_ids"].to(device)       # shape: [1, seq_len]
    attention_mask = encoded["attention_mask"].to(device)  # shape: [1, seq_len]

    # 2. Forward pass
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits  = outputs.logits  # shape: [1, 2]

    # 3. Compute probabilities with softmax
    probs = F.softmax(logits, dim=-1).squeeze().tolist()
    # DistilBERT‐SST2: label 0 = NEGATIVE, label 1 = POSITIVE
    neg_prob = probs[0]
    pos_prob = probs[1]
    score    = pos_prob - neg_prob

    return {"neg_prob": neg_prob, "pos_prob": pos_prob, "score": score}


if __name__ == "__main__":
    examples = [
        "I absolutely love this! It's fantastic and makes me so happy!",
        "This is terrible. I hated every second of it.",
        "I'm not sure how I feel about this one—it's okay, but a bit dull.",
        "Wow, that was unexpectedly amazing!",
    ]

    for sentence in examples:
        result = analyze_sentence(sentence)
        print(f"Sentence: {sentence}")
        print(f"  Negative probability: {result['neg_prob']:.4f}")
        print(f"  Positive probability: {result['pos_prob']:.4f}")
        print(f"  Transformer score  : {result['score']:+.4f}\n")