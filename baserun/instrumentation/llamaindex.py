import json
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Union,
)

import llama_index.core.instrumentation as instrument
from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from llama_index.core.instrumentation.events import BaseEvent
from llama_index.core.instrumentation.events.retrieval import RetrievalEndEvent, RetrievalStartEvent

from baserun.instrumentation.instrumentation import Instrumentation

if TYPE_CHECKING:
    from baserun.baserun import _Baserun


class BaserunLlamaEventHandler(BaseEventHandler):
    log_func: Callable[[str, Union[str, Dict]], Any]
    enabled: bool = False

    def enable(self) -> None:
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False

    @classmethod
    def class_name(cls) -> str:
        return "BaserunLlamaEventHandler"

    def handle_retrieval_start(self, event: RetrievalStartEvent) -> None:
        self.log_func("Query for nodes retrieval", event.str_or_query_bundle.to_json())

    def handle_retrieval_end(self, event: RetrievalEndEvent) -> None:
        self.log_func("Selected nodes", json.dumps([node.dict() for node in event.nodes]))

    def handle(self, event: BaseEvent, **kwargs) -> None:
        if not self.enabled:
            return

        if isinstance(event, RetrievalStartEvent):
            self.handle_retrieval_start(event)
        if isinstance(event, RetrievalEndEvent):
            self.handle_retrieval_end(event)


class LLamaIndexInstrumentation(Instrumentation):
    def __init__(self, baserun_inst: "_Baserun") -> None:
        super().__init__(baserun_inst)
        self.event_handler = BaserunLlamaEventHandler(log_func=baserun_inst.log)
        dispatcher = instrument.get_dispatcher()
        dispatcher.add_event_handler(self.event_handler)

    def instrument(self) -> None:
        self.event_handler.enable()

    def uninstrument(self) -> None:
        self.event_handler.disable()
