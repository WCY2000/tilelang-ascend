# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from tilelang.utils.npu_arch import AscendArch, get_arch_obj
from .hint import Hint
from .node import OutputNode
from typing import List
from tvm.tir import PrimFunc


def auto_infer_current_arch() -> AscendArch:
    return get_arch_obj()


@dataclass
class BaseTemplate(ABC):
    _arch: AscendArch = field(default_factory=auto_infer_current_arch, init=False, repr=False)
    _func: PrimFunc = field(default=None, init=False, repr=False)
    _output_nodes: List[OutputNode] = field(default=None, init=False, repr=False)

    @abstractmethod
    def get_hardware_aware_configs(self, arch: AscendArch = None, topk: int = 10) -> List[Hint]:
        pass

    def with_arch(self, arch: AscendArch) -> "BaseTemplate":
        self._arch = arch
        return self

    def has_arch(self) -> bool:
        return self._arch is not None

    def equivalent_function(self) -> PrimFunc:
        return self._func

    def initialize_function(self) -> None:
        raise NotImplementedError("initialize_function is not implemented")

    def set_function(self, func: PrimFunc) -> "BaseTemplate":
        self._func = func
        return self

    def set_output_nodes(self, output_nodes: List[OutputNode]) -> "BaseTemplate":
        self._output_nodes = output_nodes
        return self

    def recommend_hints(self, topk: int = 10) -> List[Hint]:
        return self.get_hardware_aware_configs(self._arch, topk)

    @property
    def arch(self) -> AscendArch:
        return self._arch

    @property
    def output_nodes(self) -> List[OutputNode]:
        return self._output_nodes

    def __post_init__(self):
        self.initialize_function()