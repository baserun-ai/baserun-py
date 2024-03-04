from abc import ABC
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from baserun.baserun import _Baserun


class Instrumentation(ABC):
    def __init__(self, baserun_inst: "_Baserun") -> None:
        self.baserun = baserun_inst

    def instrument(self) -> None:
        ...

    def uninstrument(self) -> None:
        ...
