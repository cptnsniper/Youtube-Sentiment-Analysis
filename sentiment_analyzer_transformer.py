#!/usr/bin/env python3
"""
compute_transformer_sentiment_exact.py

1. Loads 'youtube_stratified_sample.csv' (must already exist).
2. Splits each transcript into <= (model_max_length-2)-token chunks (manually handling special tokens).
3. Uses Hugging Face's distilbert-base-uncased-finetuned-sst-2-english tokenizer + model
   to get NEG/POS logits → probabilities for each chunk.
4. Averages those probabilities across chunks to produce:
     - transformer_neg_prob (average NEG prob)
     - transformer_pos_prob (average POS prob)
     - transformer_score    (pos_prob_avg - neg_prob_avg)
5. Saves results to 'youtube_with_transformer_sentiment.csv'.
"""

import os
import pandas as pd
import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ─── CONFIG ───────────────────────────────────────────────────────────────────
INPUT_CSV  = "youtube_stratified_sample.csv"
OUTPUT_CSV = "youtube_with_transformer_sentiment.csv"
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

if not os.path.isfile(INPUT_CSV):
    raise FileNotFoundError(f"Expected '{INPUT_CSV}' in this folder.")

# 1. Load scraped data
df = pd.read_csv(INPUT_CSV)
df = df.dropna(subset=["transcript"]).reset_index(drop=True)

# 2. Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Put model in eval mode (no dropout) and on CPU
model.eval()
device = torch.device("cpu")
model.to(device)

# 3. Determine chunk sizes
max_model_len = tokenizer.model_max_length  # typically 512
chunk_size     = max_model_len - 2          # reserve 2 IDs for [CLS] & [SEP]

def transcript_to_id_chunks(text: str, chunk_size: int) -> list[list[int]]:
    """
    Tokenize `text` (no special tokens), then split the token IDs
    into sublists each of length <= chunk_size.
    Returns a list of lists of token IDs (no special tokens).
    """
    # 1) Encode without special tokens
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    # 2) Slice into raw ID chunks of size == chunk_size
    chunks = [token_ids[i : i + chunk_size] for i in range(0, len(token_ids), chunk_size)]
    return chunks

# 4. Process each transcript
all_neg_probs = []
all_pos_probs = []
all_scores    = []

for idx, row in df.iterrows():
    transcript = str(row["transcript"])
    id_chunks  = transcript_to_id_chunks(transcript, chunk_size)
    
    # For each chunk, build input_ids = [CLS] + chunk_ids + [SEP], then run model
    chunk_neg = []
    chunk_pos = []
    
    for ids in id_chunks:
        # build input IDs with special tokens
        input_ids = [tokenizer.cls_token_id] + ids + [tokenizer.sep_token_id]
        input_ids_tensor = torch.tensor([input_ids], device=device)
        attention_mask   = torch.ones_like(input_ids_tensor)  # all tokens attended
        
        # forward pass, get logits (shape: [1, 2])
        with torch.no_grad():
            outputs = model(input_ids=input_ids_tensor, attention_mask=attention_mask)
            logits = outputs.logits  # shape (1, 2)
        
        # convert logits to probabilities
        probs = F.softmax(logits, dim=-1).squeeze().tolist()
        # For distilbert-sst2: label 0 = NEGATIVE, label 1 = POSITIVE
        neg_prob = probs[0]
        pos_prob = probs[1]
        
        chunk_neg.append(neg_prob)
        chunk_pos.append(pos_prob)
    
    # Average across chunks
    avg_neg = sum(chunk_neg) / len(chunk_neg)
    avg_pos = sum(chunk_pos) / len(chunk_pos)
    score   = avg_pos - avg_neg
    
    all_neg_probs.append(avg_neg)
    all_pos_probs.append(avg_pos)
    all_scores.append(score)

# 5. Attach to DataFrame and save
df["transformer_neg_prob"] = all_neg_probs
df["transformer_pos_prob"] = all_pos_probs
df["transformer_score"]    = all_scores

df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Saved '{OUTPUT_CSV}' with columns: transformer_neg_prob, transformer_pos_prob, transformer_score")