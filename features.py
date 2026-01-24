"""features.py

Sklearn-style transformer to extract emoji features from text using the lexicon in Emoji_Sentiment_Data_v1.0.csv.
"""
from __future__ import annotations
import math
from typing import List
from sklearn.base import BaseEstimator, TransformerMixin
from emoji_sentiment import EmojiSentiment


class EmojiFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract numeric emoji features for each input string.

    Features returned (per sample):
      - emoji_count: number of emoji characters found
      - unique_emoji_count
      - avg_score: average numeric_score across emojis (unweighted)
      - weighted_score: occurrences-weighted average using log(occ+1)
      - positive_ratio, neutral_ratio, negative_ratio: mean of proportions across emojis
    """

    def __init__(self, csv_path: str = 'Emoji_Sentiment_Data_v1.0.csv') -> None:
        self.csv_path = csv_path
        self.es = EmojiSentiment(csv_path)

    def fit(self, X: List[str], y=None):
        return self

    def transform(self, X: List[str]):
        rows = []
        for text in X:
            emojis = self.es.extract_emojis(text)
            unique = set(emojis)
            emoji_count = len(emojis)
            unique_count = len(unique)
            if emoji_count == 0:
                rows.append([0, 0, 0.0, 0.0, 0.0, 0.0, 0.0])
                continue

            scores = []
            weighted_sum = 0.0
            total_weight = 0.0
            pos_ratios = []
            neu_ratios = []
            neg_ratios = []
            for e in emojis:
                info = self.es.lexicon.get(e)
                if not info:
                    continue
                s = info.numeric_score()
                scores.append(s)
                w = math.log1p(info.occurrences) if info.occurrences > 0 else 1.0
                weighted_sum += s * w
                total_weight += w
                neg, neu, pos = info.proportions()
                pos_ratios.append(pos)
                neu_ratios.append(neu)
                neg_ratios.append(neg)

            avg_score = sum(scores) / (len(scores) or 1)
            weighted_score = weighted_sum / (total_weight or 1)
            pos_ratio = sum(pos_ratios) / (len(pos_ratios) or 1)
            neu_ratio = sum(neu_ratios) / (len(neu_ratios) or 1)
            neg_ratio = sum(neg_ratios) / (len(neg_ratios) or 1)

            rows.append([emoji_count, unique_count, avg_score, weighted_score, pos_ratio, neu_ratio, neg_ratio])

        return rows
