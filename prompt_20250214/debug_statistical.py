# SYSTEM IMPORTS
from collections.abc import Sequence
from pprint import pprint
import argparse as ap
import numpy as np
import os
import sys


_cd_: str = os.path.abspath(os.path.dirname(__file__))
if _cd_ not in sys.path:
    sys.path.append(_cd_)
del _cd_


# PYTHON PROJECT IMPORTS
from data.charloader import load_chars_from_file
from models.ngram.ngram import Ngram
from vocab import START_TOKEN, END_TOKEN


def main() -> None:
    train_data: Sequence[Sequence[str]] = [["the", "cat", "sat", "on", "the", "mat"]]
    dev_data: Sequence[Sequence[str]] = [["the", "cat", "sat", "on", "the", "mat"]]

    m: Ngram = Ngram(2, train_data)
    pprint(m.gram_2_logprobs)

    num_correct: int = 0
    total: int = 0
    for dev_line in dev_data:
        q = m.start()

        for c_input, c_actual in zip([START_TOKEN] + dev_line, dev_line + [END_TOKEN]):
            q, p = m.step(q, m.vocab.numberize(c_input))
            c_predicted = m.vocab.denumberize(p.argmax())

            num_correct += int(c_predicted == c_actual)
            total += 1

    print(num_correct, total, "=>", num_correct / total)

if __name__ == "__main__":
    main()

