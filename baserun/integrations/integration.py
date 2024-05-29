from abc import ABC
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from baserun.mixins import ClientMixin


class Integration(ABC):
    def __init__(self, client: "ClientMixin") -> None:
        self.client = client

    def instrument(self) -> None: ...

    def uninstrument(self) -> None: ...
