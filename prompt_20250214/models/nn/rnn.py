# SYSTEM IMPORTS
from collections.abc import Sequence, Mapping
from typing import Type, Tuple
import collections, math, random, sys
import os
import sys
import torch as pt
from tqdm import tqdm

_cd_: str = os.path.abspath(os.path.dirname(__file__))
for _dir_ in [_cd_, os.path.join(_cd_, "..", "..")]:
    if _dir_ not in sys.path:
        sys.path.append(_dir_)
del _cd_

# PYTHON PROJECT IMPORTS
from lm import LM, StateType
from vocab import Vocab, START_TOKEN, END_TOKEN

RNNType: Type = Type["RNN"]


class RNN(pt.nn.Module):
    def __init__(self: RNNType,
                 data: Sequence[Sequence[str]],
                 saved_model_path: str = None,
                 num_epochs: int = 2
                 ) -> None:
        super().__init__()
        self.vocab = Vocab()
        for line in data:
            for w in list(line) + [END_TOKEN]:
                self.vocab.add(w)

        # Hyperparameters
        self.hidden_size = 128
        self.embedding_dim = 64

        # Model layers
        self.embedding = pt.nn.Embedding(len(self.vocab), self.embedding_dim)
        self.rnn_cell = pt.nn.RNNCell(self.embedding_dim, self.hidden_size)
        self.output_layer = pt.nn.Linear(self.hidden_size, len(self.vocab))

        if saved_model_path is None:
            optimizer = pt.optim.Adam(self.parameters(), lr=1e-3)

            for epoch in range(num_epochs):
                random.shuffle(data)
                for line in tqdm(data, desc=f"epoch {epoch}"):
                    loss = 0.0
                    state = self.start()
                    for c_in, c_out in zip([START_TOKEN] + line, line + [END_TOKEN]):
                        w_idx = self.vocab.numberize(c_in)
                        state, log_probs = self.step(state, w_idx)
                        target_idx = self.vocab.numberize(c_out)
                        loss -= log_probs[target_idx]

                    optimizer.zero_grad()
                    loss.backward()
                    pt.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                    optimizer.step()
        else:
            self.load_state_dict(pt.load(saved_model_path, weights_only=True))

    def forward(self: RNNType) -> pt.Tensor:
        # Dummy implementation to comply with API
        hidden = pt.zeros(1, self.hidden_size)
        logits = self.output_layer(hidden)
        return pt.log_softmax(logits, dim=1).squeeze(0)

    def start(self: RNNType) -> StateType:
        return pt.zeros(1, self.hidden_size)

    def step(self: RNNType,
             q: StateType,
             w_idx: int
             ) -> Tuple[StateType, pt.Tensor]:
        input_tensor = pt.tensor([w_idx], dtype=pt.long)
        embedded = self.embedding(input_tensor)
        next_hidden = self.rnn_cell(embedded, q)
        logits = self.output_layer(next_hidden)
        log_probs = pt.log_softmax(logits, dim=1).squeeze(0)
        return (next_hidden, log_probs)


LSTMType: Type = Type["LSTM"]


class LSTM(pt.nn.Module):
    def __init__(self: LSTMType,
                 data: Sequence[Sequence[str]],
                 saved_model_path: str = None,
                 num_epochs: int = 2
                 ) -> None:
        super().__init__()
        self.vocab = Vocab()
        for line in data:
            for w in list(line) + [END_TOKEN]:
                self.vocab.add(w)

        # Hyperparameters
        self.hidden_size = 128
        self.embedding_dim = 64

        # Model layers
        self.embedding = pt.nn.Embedding(len(self.vocab), self.embedding_dim)
        self.lstm_cell = pt.nn.LSTMCell(self.embedding_dim, self.hidden_size)
        self.output_layer = pt.nn.Linear(self.hidden_size, len(self.vocab))

        if saved_model_path is None:
            optimizer = pt.optim.Adam(self.parameters(), lr=1e-3)

            for epoch in range(num_epochs):
                random.shuffle(data)
                for line in tqdm(data, desc=f"epoch {epoch}"):
                    loss = 0.0
                    state = self.start()  # (hidden, cell)
                    for c_in, c_out in zip([START_TOKEN] + line, line + [END_TOKEN]):
                        w_idx = self.vocab.numberize(c_in)
                        state, log_probs = self.step(state, w_idx)
                        target_idx = self.vocab.numberize(c_out)
                        loss -= log_probs[target_idx]

                    optimizer.zero_grad()
                    loss.backward()
                    pt.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                    optimizer.step()
        else:
            self.load_state_dict(pt.load(saved_model_path, weights_only=True))

    def forward(self: LSTMType) -> pt.Tensor:
        # Dummy implementation to comply with API
        hidden = pt.zeros(1, self.hidden_size)
        logits = self.output_layer(hidden)
        return pt.log_softmax(logits, dim=1).squeeze(0)

    def start(self: LSTMType) -> StateType:
        # LSTM state is (hidden, cell)
        return (
            pt.zeros(1, self.hidden_size),
            pt.zeros(1, self.hidden_size)
        )

    def step(self: LSTMType,
             state: Tuple[pt.Tensor, pt.Tensor],
             w_idx: int
             ) -> Tuple[Tuple[pt.Tensor, pt.Tensor], pt.Tensor]:
        hidden, cell = state
        input_tensor = pt.tensor([w_idx], dtype=pt.long)
        embedded = self.embedding(input_tensor)
        new_hidden, new_cell = self.lstm_cell(embedded, (hidden, cell))
        logits = self.output_layer(new_hidden)
        log_probs = pt.log_softmax(logits, dim=1).squeeze(0)
        return (new_hidden, new_cell), log_probs

