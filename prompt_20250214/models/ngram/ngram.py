# models/ngram/ngram.py

from collections.abc import Sequence, Mapping
from typing import Type, Tuple, Optional
import collections
import math
import os
import sys

_cd_ = os.path.abspath(os.path.dirname(__file__))
for _dir_ in [_cd_, os.path.join(_cd_, "..", "..")]:
    if _dir_ not in sys.path:
        sys.path.append(_dir_)
del _cd_

# PYTHON PROJECT IMPORTS
from lm import LM, StateType
from vocab import Vocab, START_TOKEN, END_TOKEN

NgramType: Type = Type["Ngram"]


class Ngram(LM):
    """
    An N-gram language model that follows the same interface
    as your unigram code. We store states as tuples of integer
    token indices, and step(...) expects an integer w_idx.
    """

    def __init__(self: NgramType,
                 N: int,
                 data: Sequence[Sequence[str]]) -> None:
        self.N = N
        self.vocab = Vocab()

        # 1) Build vocab so we have indices for all tokens, plus BOS/EOS
        for line in data:
            for token in line:
                self.vocab.add(token)
            self.vocab.add(END_TOKEN)
        self.vocab.add(START_TOKEN)

        self.bos_idx = self.vocab.numberize(START_TOKEN)
        self.eos_idx = self.vocab.numberize(END_TOKEN)

        # 2) If N=1, the start state is None. Otherwise, it's (bos_idx, bos_idx, ...)
        if N == 1:
            self.START: Optional[Tuple[int, ...]] = None
        else:
            self.START = tuple([self.bos_idx] * (N - 1))

        # 3) Count n-grams
        self.count_n = collections.Counter()  # (next_token_idx, history) -> count
        self.count_sum = collections.Counter() # history -> total
        self.unigram_count = collections.Counter()
        self.total_unigrams = 0

        numeric_data = []
        for line in data:
            # Convert line to integer indices
            idx_line = [self.vocab.numberize(tok) for tok in line]
            idx_line.append(self.eos_idx)   # append <EOS>
            numeric_data.append(idx_line)

        if N == 1:
            for idx_line in numeric_data:
                for w_idx in idx_line:
                    self.unigram_count[w_idx] += 1
                    self.total_unigrams += 1
        else:
            # For each line, pad with (N-1) bos_idx at the front
            for idx_line in numeric_data:
                padded = [self.bos_idx] * (N - 1) + idx_line
                for i in range(len(idx_line)):
                    history = tuple(padded[i : i + N - 1])
                    next_tok = padded[i + N - 1]
                    self.count_n[(next_tok, history)] += 1
                    self.count_sum[history] += 1

                    # Also track unigrams for fallback
                    self.unigram_count[next_tok] += 1
                    self.total_unigrams += 1

        # 4) Build logprob tables
        self.vocab_size = len(self.vocab)
        self.uni_logprob = {}
        for w_idx in range(self.vocab_size):
            c = self.unigram_count[w_idx]
            self.uni_logprob[w_idx] = math.log((c + 1) / (self.total_unigrams + self.vocab_size))

        self.logprob_n = {}
        if N > 1:
            # For each (next_idx, hist), p = count(hist, next_idx) / count(hist)
            for (next_idx, hist), c in self.count_n.items():
                self.logprob_n[(next_idx, hist)] = math.log(
                    c / self.count_sum[hist]
                )

    def start(self: NgramType) -> StateType:
        """
        Return the initial state. None if N=1, else tuple of bos_idx repeated (N-1) times.
        """
        return self.START

    def step(self: NgramType,
             q: StateType,
             w_idx: int
             ) -> Tuple[StateType, Mapping[str, float]]:
        """
        q: the old state (None or tuple of indices).
        w_idx: the newly-seen token index.

        Return: (r, p)
          r = the new state (None or tuple of indices).
          p = dict of {token_string: log_prob} for the next token.
        """
        # --- Compute new state r
        if self.N == 1:
            r = None
        else:
            if q is None:
                # fallback if step() called before start()
                r = tuple([self.bos_idx] * (self.N - 2) + [w_idx])
            else:
                # shift left, append w_idx
                old_tuple = tuple(q)
                if len(old_tuple) == self.N - 1:
                    r = old_tuple[1:] + (w_idx,)
                else:
                    r = old_tuple + (w_idx,)

        # --- Build p
        p = {}
        # For N=1: we have a single "unigram" distribution
        if self.N == 1:
            # Convert each vocab idx -> string
            for idx in range(self.vocab_size):
                tok_str = self.vocab.denumberize(idx)
                p[tok_str] = self.uni_logprob[idx]
            return (r, p)

        # For N>1, we look up (x, r) in self.logprob_n if we have it, else fallback
        if r not in self.count_sum or self.count_sum[r] == 0:
            # fallback to unigrams
            for idx in range(self.vocab_size):
                tok_str = self.vocab.denumberize(idx)
                p[tok_str] = self.uni_logprob[idx]
        else:
            # partial fallback approach
            for idx in range(self.vocab_size):
                if (idx, r) in self.logprob_n:
                    p[self.vocab.denumberize(idx)] = self.logprob_n[(idx, r)]
                else:
                    p[self.vocab.denumberize(idx)] = self.uni_logprob[idx]

        return (r, p)
