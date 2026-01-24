"""emoji_sentiment.py

Lightweight emoji sentiment lookup and aggregator.

Usage:
  from emoji_sentiment import EmojiSentiment
  es = EmojiSentiment('Emoji_Sentiment_Data_v1.0.csv')
  score, label, details = es.predict("I love this 😂❤")

CLI:
  python emoji_sentiment.py "text with emojis"

"""
from __future__ import annotations
import csv
import math
import sys
from dataclasses import dataclass
from typing import Dict, Tuple, List


@dataclass
class EmojiInfo:
    char: str
    occurrences: int
    negative: int
    neutral: int
    positive: int

    @property
    def total(self) -> int:
        return self.negative + self.neutral + self.positive

    def proportions(self) -> Tuple[float, float, float]:
        t = self.total or 1
        return (self.negative / t, self.neutral / t, self.positive / t)

    def numeric_score(self) -> float:
        """Return a numeric sentiment score in [-1, 1].

        We compute (pos - neg) / total.
        """
        t = self.total or 1
        return (self.positive - self.negative) / t


class EmojiSentiment:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.lexicon: Dict[str, EmojiInfo] = {}
        self._load()

    def _load(self):
        with open(self.csv_path, encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                char = row.get('Emoji')
                if not char:
                    continue
                try:
                    occ = int(row.get('Occurrences', '0') or 0)
                except ValueError:
                    occ = 0
                try:
                    neg = int(row.get('Negative', '0') or 0)
                except ValueError:
                    neg = 0
                try:
                    neu = int(row.get('Neutral', '0') or 0)
                except ValueError:
                    neu = 0
                try:
                    pos = int(row.get('Positive', '0') or 0)
                except ValueError:
                    pos = 0
                self.lexicon[char] = EmojiInfo(char, occ, neg, neu, pos)

    def extract_emojis(self, text: str) -> List[str]:
        # Find characters in text that are keys in the lexicon
        return [ch for ch in text if ch in self.lexicon]

    def predict(self, text: str) -> Tuple[float, str, Dict[str, object]]:
        """Predict aggregated emoji sentiment for given text.

        Returns (score, label, details) where score is in [-1,1], label one of 'negative','neutral','positive'.
        details contains per-emoji breakdown.
        """
        emojis = self.extract_emojis(text)
        if not emojis:
            return 0.0, 'neutral', {'emojis': [], 'per_emoji': {}}

        total_weight = 0.0
        weighted_sum = 0.0
        per = {}
        for e in emojis:
            info = self.lexicon.get(e)
            if not info:
                continue
            weight = math.log1p(info.occurrences) if info.occurrences > 0 else 1.0
            s = info.numeric_score()
            weighted_sum += s * weight
            total_weight += weight
            per[e] = {
                'score': s,
                'occurrences': info.occurrences,
                'proportions': info.proportions(),
            }

        score = weighted_sum / (total_weight or 1.0)

        # label thresholds: >0.1 pos, <-0.1 neg, else neutral (tunable)
        if score > 0.1:
            label = 'positive'
        elif score < -0.1:
            label = 'negative'
        else:
            label = 'neutral'

        return score, label, {'emojis': emojis, 'per_emoji': per}


def _cli(argv):
    if len(argv) < 2:
        print('Usage: python emoji_sentiment.py "text with emojis"')
        return 2
    text = argv[1]
    es = EmojiSentiment('Emoji_Sentiment_Data_v1.0.csv')
    score, label, details = es.predict(text)
    print(f'Text: {text}')
    print(f'Sentiment score: {score:.4f}  label: {label}')
    print('Details:')
    for e, d in details['per_emoji'].items():
        neg, neu, pos = d['proportions']
        print(f'  {e} -> score={d["score"]:.3f}, occ={d["occurrences"]}, proportions=(neg={neg:.2f},neu={neu:.2f},pos={pos:.2f})')
    return 0


if __name__ == '__main__':
    sys.exit(_cli(sys.argv))
