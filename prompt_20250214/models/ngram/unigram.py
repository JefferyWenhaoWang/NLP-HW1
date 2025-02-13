# SYSTEM IMPORTS
from collections.abc import Sequence, Mapping
from typing import Type, Tuple
import collections
import numpy as np
import os
import sys


_cd_: str = os.path.abspath(os.path.dirname(__file__))
for _dir_ in [_cd_, os.path.join(_cd_, "..", "..")]:
    if _dir_ not in sys.path:
        sys.path.append(_dir_)
del _cd_


# PYTHON PROJECT IMPORTS
from lm import LM, StateType
from vocab import Vocab, START_TOKEN, END_TOKEN


# Types declared in this module
UnigramType: Type = Type["Unigram"]


class Unigram(LM):
    """A unigram language model.

    data: a list of lists of symbols. They should not contain '<EOS>' or '<BOS>';
          the '<EOS>' symbol is automatically appended during
          training.
    """
    
    def __init__(self: UnigramType,
                 data: Sequence[Sequence[str]]
                 ) -> None:
        self.vocab: Vocab = Vocab()
        count: collections.Counter = collections.Counter()
        total: int = 0
        for line in data:
            for w in list(line) + [END_TOKEN]:
                self.vocab.add(w)

                w_idk: int = self.vocab.numberize(w)
                count[w_idk] += 1
                total += 1

        self.logprob: np.ndarray = np.zeros(len(self.vocab), dtype=float)
        for w_idx in range(len(self.vocab)):
            self.logprob[w_idx] = np.log(count[w_idx]/total) if count[w_idx] > 0 else -np.inf

    def start(self: UnigramType) -> StateType:
        """Return the language model's start state. (A unigram model doesn't
        have a state, so it's just `None`."""
        
        return None

    def step(self: UnigramType,
             q: StateType,
             w_idx: int
             ) -> Tuple[StateType, Mapping[str, float]]:
        """Compute one step of the language model.

        Arguments:
        - q: The current state of the model
        - w_idk: The most recently seen numberized token (int)

        Return: (r, pb), where
        - r: The state of the model after reading 'w_idk'
        - pb: The log-probability distribution over the next token (after reading 'w_idx')
        """
        
        return (None, self.logprob)

