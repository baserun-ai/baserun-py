import json
from typing import (
    Any,
    Callable,
    Union,
)

import llama_index.core.instrumentation as instrument
from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from llama_index.core.instrumentation.events import BaseEvent
from llama_index.core.instrumentation.events.retrieval import RetrievalEndEvent, RetrievalStartEvent

from baserun.integrations.integration import Integration
from baserun.mixins import ClientMixin


class BaserunLlamaEventHandler(BaseEventHandler):
    log_func: Callable[[str, Union[str]], Any]
    enabled: bool = False

    def enable(self) -> None:
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False

    @classmethod
    def class_name(cls) -> str:
        return "BaserunLlamaEventHandler"

    def handle_retrieval_start(self, event: RetrievalStartEvent) -> None:
        if isinstance(event.str_or_query_bundle, str):
            self.log_func(event.str_or_query_bundle, "Query for nodes retrieval")
        else:
            self.log_func(event.str_or_query_bundle.to_json(), "Query for nodes retrieval")

    def handle_retrieval_end(self, event: RetrievalEndEvent) -> None:
        self.log_func(json.dumps([node.dict() for node in event.nodes]), "Selected nodes")

    def handle(self, event: BaseEvent, **kwargs) -> None:
        if not self.enabled:
            return

        if isinstance(event, RetrievalStartEvent):
            self.handle_retrieval_start(event)
        if isinstance(event, RetrievalEndEvent):
            self.handle_retrieval_end(event)


class LLamaIndexInstrumentation(Integration):
    def __init__(self, client: ClientMixin) -> None:
        super().__init__(client=client)
        self.event_handler = BaserunLlamaEventHandler(log_func=client.log)
        dispatcher = instrument.get_dispatcher()
        dispatcher.add_event_handler(self.event_handler)

    def instrument(self) -> None:
        self.event_handler.enable()

    def uninstrument(self) -> None:
        self.event_handler.disable()
