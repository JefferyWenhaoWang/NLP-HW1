# experiment_ngram.py

from collections.abc import Sequence
from typing import Tuple
import os
import sys

_cd_ = os.path.abspath(os.path.dirname(__file__))
if _cd_ not in sys.path:
    sys.path.append(_cd_)
del _cd_

from data.charloader import load_chars_from_file
from models.ngram.ngram import Ngram
from vocab import START_TOKEN, END_TOKEN


def train_ngram(N: int) -> Ngram:
    """
    Trains an N-gram model on data/large and returns the model.
    """
    train_data: Sequence[Sequence[str]] = load_chars_from_file("./data/large")
    m = Ngram(N, train_data)
    return m


def dev_ngram(m: Ngram) -> Tuple[int, int]:
    """
    Evaluate Ngram model on data/dev. Return (num_correct, total).
    """
    dev_data: Sequence[Sequence[str]] = load_chars_from_file("./data/dev")
    num_correct = 0
    total = 0

    for line in dev_data:
        # We'll predict each next character in line
        q = m.start()

        # We'll read one token at a time
        for i in range(len(line)):
            w_idx = m.vocab.numberize(line[i])  # pass the int index
            q, p = m.step(q, w_idx)

            # The "correct" next char is line[i+1], if exists, else <EOS>
            if i + 1 < len(line):
                actual_char = line[i+1]
            else:
                actual_char = END_TOKEN

            # predicted token is the one with highest log-prob in p
            predicted_char = max(p.keys(), key=lambda x: p[x])
            if predicted_char == actual_char:
                num_correct += 1
            total += 1

    return num_correct, total


def test_ngram(m: Ngram) -> Tuple[int, int]:
    """
    Evaluate on data/test. Return (num_correct, total).
    """
    test_data: Sequence[Sequence[str]] = load_chars_from_file("./data/test")
    num_correct = 0
    total = 0

    for line in test_data:
        q = m.start()
        for i in range(len(line)):
            w_idx = m.vocab.numberize(line[i])
            q, p = m.step(q, w_idx)

            if i + 1 < len(line):
                actual_char = line[i+1]
            else:
                actual_char = END_TOKEN

            predicted_char = max(p.keys(), key=lambda x: p[x])
            if predicted_char == actual_char:
                num_correct += 1
            total += 1

    return num_correct, total


def main():
    m = train_ngram(5)
    dev_correct, dev_total = dev_ngram(m)
    test_correct, test_total = test_ngram(m)
    print(f"dev acc: {dev_correct/dev_total:.4f}")
    print(f"test acc: {test_correct/test_total:.4f}")


if __name__ == "__main__":
    main()
