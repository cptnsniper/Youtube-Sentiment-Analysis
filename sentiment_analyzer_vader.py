#!/usr/bin/env python3
"""
compute_sentiment.py

Reads your stratified sample CSV, computes VADER sentiment scores
(neg/neu/pos/compound) on each full transcript, and writes out
a new CSV with those four extra columns.
"""

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# 1. Load your scraped data
df = pd.read_csv('channel_videos.csv')

# 2. Drop any rows where transcript is missing
df = df.dropna(subset=['transcript'])

# 3. Initialize the VADER analyzer once
analyzer = SentimentIntensityAnalyzer()

# 4. Apply to each transcript, expand into four columns
scores = df['transcript'].apply(lambda txt: analyzer.polarity_scores(str(txt)))
sentiment_df = pd.DataFrame(scores.tolist())

# 5. Merge and save
df = pd.concat([df, sentiment_df], axis=1)
df.to_csv('youtube_with_sentiment.csv', index=False)

print("✅ Saved youtube_with_sentiment.csv with columns: ", list(sentiment_df.columns))