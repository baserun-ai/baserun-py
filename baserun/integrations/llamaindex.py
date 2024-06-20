import json
import logging
from datetime import datetime
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Union,
)
from uuid import uuid4

import llama_index.core.instrumentation as instrument
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from llama_index.core.instrumentation.events import BaseEvent
from llama_index.core.instrumentation.events.llm import LLMChatEndEvent, LLMChatStartEvent
from llama_index.core.instrumentation.events.query import QueryStartEvent
from llama_index.core.instrumentation.events.retrieval import RetrievalEndEvent, RetrievalStartEvent

from baserun.integrations.integration import Integration
from baserun.mixins import ClientMixin
from baserun.models.tags import Log
from baserun.wrappers.generic import (
    GenericChoice,
    GenericClient,
    GenericCompletion,
    GenericCompletionMessage,
    GenericInputMessage,
    GenericUsage,
)

logger = logging.getLogger(__name__)


class BaserunLlamaEventHandler(BaseEventHandler):
    client: ClientMixin
    log_func: Callable[[str, Union[str]], Any]
    enabled: bool = False
    last_llm_start: Optional[datetime] = None

    def enable(self) -> None:
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False

    @classmethod
    def class_name(cls) -> str:
        return "BaserunLlamaEventHandler"

    @staticmethod
    def _message_dict(message_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not message_data:
            return {}

        message_kwargs = message_data.get("additional_kwargs", {})

        role: MessageRole = message_data.get("role", MessageRole.USER)
        return_kwargs = {
            "content": message_data.get("content"),
            "role": role.lower(),
        }
        if message_kwargs.get("name"):
            return_kwargs["name"] = message_kwargs.get("name")
        if message_kwargs.get("tool_call_id"):
            return_kwargs["tool_call_id"] = message_kwargs.get("tool_call_id")
        if message_kwargs.get("tool_calls"):
            return_kwargs["tool_calls"] = [
                call if isinstance(call, dict) else call.dict() for call in message_kwargs.get("tool_calls", [])
            ]

        return return_kwargs

    @staticmethod
    def _guess_llm_step(response_message: ChatMessage, input_messages: List[ChatMessage]) -> str:
        """Given the LLM call input/response, we do our best to guess which step it was. We don't have
        any context from the event itself, so we have to guess based on the messages and the tool calls.

        This is pretty awkward because the objects in the response_message are classes from the LLM library,
        not Llama. (Meaning that the functions/properties on the messages aren't standard)
        """
        tool_call = response_message.additional_kwargs.get("tool_calls", [])[0]
        if (
            tool_call
            and hasattr(tool_call, "function")
            and hasattr(tool_call.function, "name")
            and tool_call.function.name == "query_engine_tool"
        ):
            return "Condense Query"

        if input_messages[0].role == MessageRole.SYSTEM and not response_message.additional_kwargs:
            return "Synthesize"

        if input_messages[-1].role == MessageRole.TOOL:
            return "Generate Answer"

        # TODO: Figure out other step names, e.g. other chat modes (above are just for the `best` mode)
        return "Generation"

    def handle_llm_start(self, event: LLMChatStartEvent) -> None:
        self.last_llm_start = event.timestamp

    def handle_llm_end(self, event: LLMChatEndEvent) -> None:
        start_timestamp = self.last_llm_start or event.timestamp
        input_messages = []
        for message in event.messages:
            input_message = GenericInputMessage(
                **self._message_dict(message.dict()),
            )
            input_messages.append(input_message)

        if not event.response or not event.response.raw:
            return

        raw_response = event.response.raw
        response_message = event.response.message
        raw_choice = raw_response.get("choices", [])[0]
        choice = GenericChoice(
            finish_reason=raw_choice.finish_reason if raw_choice else "stop",
            logprobs=raw_choice.logprobs if raw_choice else None,
            index=raw_choice.index if raw_choice else 0,
            message=GenericCompletionMessage(
                **self._message_dict(response_message.dict()),
            ),
        )
        raw_usage = raw_response.get("usage")
        if raw_usage:
            usage = GenericUsage(**raw_usage.dict())
        else:
            usage = GenericUsage(
                completion_tokens=0,
                prompt_tokens=0,
                total_tokens=0,
            )

        name = self._guess_llm_step(response_message, event.messages)

        completion = GenericCompletion(
            name=name,
            model=raw_response.get("model", "llama"),
            id=raw_response.get("id"),
            client=self.client,
            # FIXME: Why is default_factory in the BaseModel not working?
            completion_id=str(uuid4()),
            trace_id=self.client.trace_id,
            input_messages=input_messages,
            choices=[choice],
            usage=usage,
            start_timestamp=start_timestamp,
            end_timestamp=event.timestamp,
        )

        completion.submit_to_baserun()

    def handle_query_start(self, event: QueryStartEvent) -> Log:
        # Note that there are (at least) two query start events: one before the condense query and one before the
        # retrieval query. We can't distinguish between the two
        if isinstance(event.query, str):
            return self.log_func(json.dumps({"input": event.query}), "Query Start")

        return self.log_func(str(event.query), "Query Start")

    def handle_retrieval_start(self, event: RetrievalStartEvent) -> Log:
        if isinstance(event.str_or_query_bundle, str):
            return self.log_func(event.str_or_query_bundle, "Retrieval query")
        else:
            return self.log_func(event.str_or_query_bundle.to_json(), "Retrieval query")

    def handle_retrieval_end(self, event: RetrievalEndEvent) -> Log:
        return self.log_func(json.dumps([node.dict() for node in event.nodes]), "Selected nodes")

    def handle(self, event: BaseEvent, **kwargs) -> None:
        if not self.enabled:
            return

        log = None
        if isinstance(event, QueryStartEvent):
            log = self.handle_query_start(event)
        elif isinstance(event, RetrievalStartEvent):
            log = self.handle_retrieval_start(event)
        elif isinstance(event, RetrievalEndEvent):
            log = self.handle_retrieval_end(event)
            self.client.submit_to_baserun()
        elif isinstance(event, LLMChatStartEvent):
            self.handle_llm_start(event)
        elif isinstance(event, LLMChatEndEvent):
            self.handle_llm_end(event)
        elif hasattr(event, "to_json"):
            log = self.log_func(event.to_json(), event.__class__.__name__)
        else:
            logger.debug(f"Unhandled event: {event.__class__}")

        if log:
            log.timestamp = event.timestamp


class LLamaIndexInstrumentation(Integration):
    def __init__(self, client: Optional[ClientMixin] = None) -> None:
        if not client:
            client = GenericClient(name="llama", autosubmit=False)

        super().__init__(client=client)
        self.event_handler = BaserunLlamaEventHandler(log_func=client.log, client=client)
        dispatcher = instrument.get_dispatcher()
        dispatcher.add_event_handler(self.event_handler)

    def instrument(self) -> None:
        self.event_handler.enable()

    def uninstrument(self) -> None:
        self.event_handler.disable()

    @classmethod
    def start(cls, client: Optional[ClientMixin] = None) -> "LLamaIndexInstrumentation":
        instrumentation = cls(client=client)
        instrumentation.instrument()
        return instrumentation
