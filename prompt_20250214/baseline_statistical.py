# SYSTEM IMPORTS
from collections.abc import Sequence
from typing import Tuple
from pprint import pprint
import argparse as ap
import numpy as np
import os
import sys


# make sure the directory that contains this file is in sys.path
_cd_: str = os.path.abspath(os.path.dirname(__file__))
if _cd_ not in sys.path:
    sys.path.append(_cd_)
del _cd_


# PYTHON PROJECT IMPORTS
from data.charloader import load_chars_from_file
from models.ngram.unigram import Unigram
from vocab import START_TOKEN, END_TOKEN


def train_unigram() -> Unigram:
    train_data: Sequence[Sequence[str]] = load_chars_from_file("./data/large")
    return Unigram(train_data)


def dev_unigram(m: Unigram) -> Tuple[int, int]:
    dev_data: Sequence[Sequence[str]] = load_chars_from_file("./data/dev")

    num_correct: int = 0
    total: int = 0
    for dev_line in dev_data:
        q = m.start()  # get the initial state of the model

        for c_input, c_actual in zip([START_TOKEN] + dev_line, # read in string w/ <BOS> prepended
                                      dev_line + [END_TOKEN]): # check against string incl. <EOS>
            q, p = m.step(q, m.vocab.numberize(c_input))
            c_predicted = m.vocab.denumberize(np.argmax(p))

            num_correct += int(c_predicted == c_actual)
            total += 1
    return num_correct, total


def test_unigram(m: Unigram) -> Tuple[int, int]:
    test_data: Sequence[Sequence[str]] = load_chars_from_file("./data/test")

    num_correct: int = 0
    total: int = 0
    for test_line in test_data:
        q = m.start()  # get the initial state of the model

        for c_input, c_actual in zip([START_TOKEN] + test_line, # read in string w/ <BOS> prepended
                                      test_line + [END_TOKEN]): # check against string incl. <EOS>
            q, p = m.step(q, m.vocab.numberize(c_input))
            c_predicted = m.vocab.denumberize(np.argmax(p))

            num_correct += int(c_predicted == c_actual)
            total += 1
    return num_correct, total


def main() -> None:
    m: Unigram = train_unigram()
    num_correct_dev, total_dev = dev_unigram(m)
    num_correct_test, total_test = test_unigram(m)

    print(f'baseline stat dev acc: {num_correct_dev / total_dev}')
    print(f'baseline stat test acc: {num_correct_test / total_test}')

if __name__ == "__main__":
    main()

