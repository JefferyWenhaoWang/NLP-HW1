# SYSTEM IMPORTS
from collections.abc import Sequence, Mapping
from abc import ABC, abstractmethod
from typing import Type, Tuple, Union
import collections
import math
import numpy as np
import torch as pt


# PYTHON PROJECT IMPORTS


# Types declared in this module
LMType: Type = Type["LM"]

StateType: Type = Union[Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],   # one np.array or a pair of them
                        Union[pt.Tensor, Tuple[pt.Tensor, pt.Tensor]]]      # one pt.Tensor or a pair of them



class LM(ABC):

    @abstractmethod
    def start(self: LMType) -> StateType:
        ...

    @abstractmethod
    def step(self: LMType,
             q: StateType,
             w_idx: int
             ) -> Tuple[StateType, Mapping[int, float]]:
        ...

