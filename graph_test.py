#!/usr/bin/env python3
"""
analyze_sentiment_flipped.py

Loads the CSV with sentiment scores, computes correlations,
and draws a scatterplot with Views on the x-axis and
Compound Sentiment on the y-axis.
"""

import pandas as pd
import matplotlib.pyplot as plt

# 1. Load augmented data
df = pd.read_csv('youtube_with_sentiment.csv')

# 2. Compute and print correlation matrix
perf_and_sent = ['views', 'likes', 'comments', 'neg', 'neu', 'pos', 'compound']
corr = df[perf_and_sent].corr()
print("Correlation matrix:")
print(corr.round(3))

# 3. Scatter: Views (x-axis) vs. Compound Sentiment (y-axis)
plt.figure()
plt.scatter(df['views'], df['compound'])
plt.xlabel('Views')
plt.ylabel('Compound Sentiment Score')
plt.title('Compound Sentiment Score vs. Views')
plt.tight_layout()

# 4. Save and show
plt.savefig('sentiment_vs_views.png')
plt.show()