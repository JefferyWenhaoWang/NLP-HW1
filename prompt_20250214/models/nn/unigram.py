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


UnigramType: Type = Type["Unigram"]


class Unigram(pt.nn.Module):
    def __init__(self: UnigramType,
                 data: Sequence[Sequence[str]],
                 saved_model_path: str = None,
                 num_epochs: int = 2
                 ) -> None:
        # Call parent class's __init__(). You will get an error
        # if you forget this.
        super().__init__() 

        # Store the vocab inside the Unigram object,
        # so when we save the Unigram, it saves the vocab too.
        self.vocab = Vocab()

        for line in data:
            for w in list(line) + [END_TOKEN]:
                self.vocab.add(w)
        
        # Create the model parameters. Wrapping it inside
        # Parameter tells Module to manage it for us, so
        # we don't have to set requires_grad.
        self.logits = pt.nn.Parameter(
                        pt.normal(mean=0, std=0.01, 
                                  size=(len(self.vocab),))
                      )

        if saved_model_path is None:
            o = pt.optim.Adam(self.parameters(), lr=1e-3)

            for epoch in range(num_epochs):

                # train the model
                random.shuffle(data) # Important

                train_chars = 0 # Total number of characters
                for line in tqdm(data, desc=f"epoch {epoch}"):
                    # Compute the negative log-likelihood of this line,
                    # which is the thing we want to minimize.
                    loss = 0.
                    q = self.start()
                    for c_in, c_out in zip([START_TOKEN] + line, line + [END_TOKEN]): # skip BOS
                        train_chars += 1
                        q, p = self.step(q, self.vocab.numberize(c_in))
                        loss -= p[self.vocab.numberize(c_out)]

                    # Compute gradient of loss with respect to parameters.
                    o.zero_grad()   # Reset gradients to zero
                    loss.backward() # Add in the gradient of loss

                    # Clip gradients (not needed here, but helpful for RNNs)
                    pt.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

                    # Do one step of gradient descent.
                    o.step()
        else:
            self.load_state_dict(pt.load(saved_model_path, weights_only=True))

    # You can define whatever methods you want,
    # but "forward" is special, so the main method should
    # be named "forward".
    def forward(self: UnigramType) -> pt.Tensor:
        return pt.log_softmax(self.logits, dim=0)

    def start(self: UnigramType) -> StateType:
        return None

    def step(self: UnigramType,
             q: StateType,
             w_idx: int
             ) -> Tuple[StateType, Mapping[str, float]]:
        return (q, self.forward())

