import json
import logging
from datetime import datetime, timezone
from types import TracebackType
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, Generic, Iterator, List, Optional, TypeVar, Union
from uuid import uuid4

import anthropic
import anthropic.resources
import httpx  # noqa: F401
from pydantic import BaseModel, ConfigDict, Field

from baserun.api import ApiClient
from baserun.integrations.integration import Integration
from baserun.mixins import ClientMixin, CompletionMixin
from baserun.models.evals import CompletionEval, TraceEval
from baserun.models.tags import Tag
from baserun.utils import copy_type_hints
from baserun.wrappers.generic import (
    GenericChoice,
    GenericClient,
    GenericCompletion,
    GenericCompletionMessage,
    GenericInputMessage,
    GenericUsage,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class WrappedMessage(anthropic.types.Message, CompletionMixin):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    client: "WrappedAnthropicBaseClient"
    name: str
    error: Optional[str] = None
    trace_id: str
    completion_id: str
    template: Optional[str]
    start_timestamp: datetime
    first_token_timestamp: datetime
    end_timestamp: datetime
    input_messages: List[anthropic.types.MessageParam] = Field(default_factory=list)
    config_params: Dict[str, Any] = Field(default_factory=dict)
    tool_results: List[Any] = Field(default_factory=list)
    tags: List[Tag] = Field(default_factory=list)
    evals: List[CompletionEval] = Field(default_factory=list)

    def genericize(self):
        choices = []
        for content in self.content:
            message = GenericCompletionMessage(role="assistant", content=content.text)
            choices.append(GenericChoice(message=message))

        input_messages = []
        for message in self.input_messages:
            content = message.get("content", None)
            generic_content = content
            if isinstance(content, Iterator):
                generic_content = json.dumps([c for c in content])

            input_messages.append(GenericInputMessage(role=message.get("role"), content=generic_content))

        usage = GenericUsage(
            completion_tokens=self.usage.output_tokens,
            prompt_tokens=self.usage.input_tokens,
            total_tokens=self.usage.output_tokens + self.usage.input_tokens,
        )

        return GenericCompletion(
            id=self.id,
            name=self.name,
            error=self.error,
            trace_id=self.trace_id,
            completion_id=self.completion_id,
            template=self.template,
            start_timestamp=self.start_timestamp,
            first_token_timestamp=self.first_token_timestamp,
            end_timestamp=self.end_timestamp,
            client=self.client.genericize(),
            choices=choices,
            usage=usage,
            request_id=self.client.request_ids.get(self.id),
            config_params=self.config_params,
            tool_results=self.tool_results,
            tags=self.tags,
            evals=self.evals,
            input_messages=input_messages,
        )

    def submit_to_baserun(self):
        self.client.api_client.submit_completion(self.genericize())


class WrappedStreamBase(CompletionMixin, BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    client: "WrappedAnthropicBaseClient"
    id: Optional[str] = None
    name: str
    error: Optional[str] = None
    trace_id: str
    completion_id: str
    template: Optional[str] = None
    start_timestamp: datetime
    end_timestamp: Optional[datetime] = None
    first_token_timestamp: Optional[datetime] = None
    input_messages: List[anthropic.types.MessageParam] = Field(default_factory=list)
    config_params: Dict[str, Any] = Field(default_factory=dict)
    tool_results: List[Any] = Field(default_factory=list)
    tags: List[Tag] = Field(default_factory=list)
    evals: List[CompletionEval] = Field(default_factory=list)

    # Used for evals on streaming responses
    captured_messages: List[anthropic.types.Message] = Field(default_factory=list)

    def __init__(self, *args, **kwargs):
        CompletionMixin.__init__(self)
        BaseModel.__init__(self, **kwargs)

    def genericize(self):
        usage = GenericUsage(
            completion_tokens=0,
            prompt_tokens=0,
            total_tokens=0,
        )

        choices = []
        for message in self.captured_messages:
            usage.completion_tokens += message.usage.output_tokens
            usage.prompt_tokens += message.usage.input_tokens
            for content in message.content:
                choices.append(GenericChoice(message=GenericCompletionMessage(content=content.text, role="assistant")))

        usage.total_tokens = usage.completion_tokens + usage.prompt_tokens

        return GenericCompletion(
            id=self.id,
            name=self.name,
            error=self.error,
            trace_id=self.trace_id,
            completion_id=self.completion_id,
            template=self.template,
            start_timestamp=self.start_timestamp,
            first_token_timestamp=self.first_token_timestamp,
            end_timestamp=self.end_timestamp,
            client=self._client.genericize(),
            choices=choices,
            usage=usage,
            request_id=self.client.request_ids.get(self.id),
            tool_results=self.tool_results,
            tags=self.tags,
            evals=self.evals,
            input_messages=self.input_messages,
        )

    def submit_to_baserun(self):
        if not self.id:
            return

        if not self.end_timestamp:
            self.end_timestamp = datetime.now(timezone.utc)
        self.client.api_client.submit_stream(self.genericize())


class WrappedAsyncStream(WrappedStreamBase, anthropic.AsyncStream, Generic[T]):
    def __init__(self, *args, **kwargs):
        WrappedStreamBase.__init__(self, *args, **kwargs)
        anthropic.AsyncStream.__init__(
            self,
            *args,
            cast_to=anthropic.types.MessageStreamEvent,
            **self._clean_kwargs(kwargs),
        )
        self._iterator = self.__stream__()

    async def __aiter__(self) -> AsyncIterator[anthropic.types.MessageStreamEvent]:  # type: ignore[override]
        async for item in self._iterator:
            yield item

    async def __stream__(self) -> AsyncIterator[anthropic.types.MessageStreamEvent]:
        captured_message = None
        content_block = None

        async for item in super().__stream__():
            if isinstance(item, anthropic.types.MessageStartEvent):
                captured_message = item.message
                self.id = item.message.id
            elif captured_message and isinstance(item, anthropic.types.ContentBlockStartEvent):
                content_block = item.content_block
                captured_message.content.append(content_block)
            elif content_block and isinstance(item, anthropic.types.ContentBlockDeltaEvent):
                content_block.text += item.delta.text
            elif captured_message and isinstance(item, anthropic.types.MessageDeltaEvent):
                captured_message.stop_reason = item.delta.stop_reason
                captured_message.stop_sequence = item.delta.stop_sequence
                captured_message.usage.output_tokens += item.usage.output_tokens
            # we don't currently care about ContentBlockStopEvent and MessageStopEvent

            yield item

        if captured_message:
            self.captured_messages.append(captured_message)
            self.submit_to_baserun()


class WrappedSyncStream(WrappedStreamBase, anthropic.Stream, Generic[T]):
    def __init__(self, *args, **kwargs):
        WrappedStreamBase.__init__(self, *args, **kwargs)
        anthropic.Stream.__init__(
            self,
            *args,
            cast_to=anthropic.types.MessageStreamEvent,
            **self._clean_kwargs(kwargs),
        )
        self._iterator = self.__stream__()

    def __iter__(self) -> Iterator[anthropic.types.MessageStreamEvent]:  # type: ignore[override]
        for item in self._iterator:
            yield item

    def __stream__(self) -> Iterator[anthropic.types.MessageStreamEvent]:
        stream = super().__stream__()
        captured_message = None
        content_block = None

        for item in stream:
            if isinstance(item, anthropic.types.MessageStartEvent):
                captured_message = item.message
                self.id = item.message.id
            elif captured_message and isinstance(item, anthropic.types.ContentBlockStartEvent):
                content_block = item.content_block
                captured_message.content.append(content_block)
            elif content_block and isinstance(item, anthropic.types.ContentBlockDeltaEvent):
                content_block.text += item.delta.text
            elif captured_message and isinstance(item, anthropic.types.MessageDeltaEvent):
                captured_message.stop_reason = item.delta.stop_reason
                captured_message.stop_sequence = item.delta.stop_sequence
                captured_message.usage.output_tokens += item.usage.output_tokens
            # we don't currently care about ContentBlockStopEvent and MessageStopEvent

            yield item

        if captured_message:
            self.captured_messages.append(captured_message)
            self.submit_to_baserun()


class WrappedMessageStreamManager(anthropic.MessageStreamManager, Generic[T]):
    def __init__(
        self,
        api_request: Callable[[], anthropic.MessageStream],
        stream: WrappedSyncStream[anthropic.types.MessageStreamEvent],
    ) -> None:
        super().__init__(api_request)
        self.__stream: WrappedSyncStream[anthropic.types.MessageStreamEvent] = stream

    def __enter__(self) -> WrappedSyncStream[anthropic.types.MessageStreamEvent]:
        return self.__stream

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self.__stream is not None:
            self.__stream.close()


class WrappedAsyncMessageStreamManager(anthropic.AsyncMessageStreamManager, Generic[T]):
    def __init__(
        self,
        stream_params: Dict[str, Any],
        api_request: Awaitable[anthropic.MessageStream],
    ) -> None:
        self.stream_params = stream_params
        super().__init__(api_request)

    async def __aenter__(self) -> WrappedAsyncStream[anthropic.types.MessageStreamEvent]:
        stream = await super().__aenter__()

        return WrappedAsyncStream(**self.stream_params, response=stream.response)


class WrappedSyncMessages(anthropic.resources.Messages):
    _client: "WrappedSyncAnthropicClient"

    @copy_type_hints(anthropic.resources.Messages)
    def create(
        self,
        *args,
        name: Optional[str] = None,
        variables: Optional[Dict[str, str]] = None,
        template: Optional[str] = None,
        **kwargs,
    ) -> Union[WrappedMessage, WrappedSyncStream[anthropic.types.MessageStreamEvent]]:
        start_timestamp = datetime.now(timezone.utc)
        stream_or_completion = super().create(*args, **kwargs)
        first_token_timestamp = datetime.now(timezone.utc)

        messages: List[anthropic.types.MessageParam] = kwargs.pop("messages", [])
        wrapped_completion: Union[WrappedSyncStream, WrappedMessage, None] = None

        if isinstance(stream_or_completion, anthropic.types.Message):
            output = stream_or_completion.content[0].text
            if not name:
                name = output[:20]

            wrapped_completion = WrappedMessage(
                client=self._client,
                trace_id=self._client.trace_id,
                name=name,
                template=template,
                start_timestamp=start_timestamp,
                end_timestamp=first_token_timestamp,
                first_token_timestamp=first_token_timestamp,
                completion_id=str(uuid4()),
                config_params=kwargs,
                input_messages=messages,
                **stream_or_completion.model_dump(),
            )
        else:
            if not name:
                name = "Anthropic Stream"

            wrapped_completion = WrappedSyncStream(
                response=stream_or_completion.response,
                client=self._client,
                trace_id=self._client.trace_id,
                name=name,
                template=template,
                start_timestamp=start_timestamp,
                end_timestamp=None,
                first_token_timestamp=first_token_timestamp,
                config_params=kwargs,
                input_messages=messages,
                completion_id=str(uuid4()),
            )

        if wrapped_completion is None:
            return stream_or_completion

        if variables:
            for key, value in variables.items():
                wrapped_completion.variable(key, value)
        else:
            wrapped_completion.submit_to_baserun()

        return wrapped_completion

    @copy_type_hints(anthropic.resources.Messages)
    def stream(
        self,
        *args,
        name: Optional[str] = None,
        variables: Optional[Dict[str, str]] = None,
        template: Optional[str] = None,
        **kwargs,
    ) -> WrappedMessageStreamManager[WrappedSyncStream]:
        if not name:
            name = "Anthropic Stream"

        start_timestamp = datetime.now(timezone.utc)
        stream_manager: anthropic.MessageStreamManager[anthropic.MessageStream] = super().stream(*args, **kwargs)
        request_callable = stream_manager._MessageStreamManager__api_request  # type: ignore[attr-defined]
        stream = request_callable()
        first_token_timestamp = datetime.now(timezone.utc)

        messages: List[anthropic.types.MessageParam] = kwargs.pop("messages", [])
        # Why is this type hint needed
        wrapped_stream: WrappedSyncStream = WrappedSyncStream(
            response=stream.response,
            client=self._client,
            trace_id=self._client.trace_id,
            name=name,
            template=template,
            start_timestamp=start_timestamp,
            end_timestamp=None,
            first_token_timestamp=first_token_timestamp,
            config_params=kwargs,
            input_messages=messages,
            completion_id=str(uuid4()),
        )

        if variables:
            for key, value in variables.items():
                wrapped_stream.variable(key, value)
        else:
            wrapped_stream.submit_to_baserun()

        return WrappedMessageStreamManager(api_request=request_callable, stream=wrapped_stream)


class WrappedAsyncMessages(anthropic.resources.AsyncMessages):
    _client: "WrappedAsyncAnthropicClient"

    @copy_type_hints(anthropic.resources.AsyncMessages)
    async def create(
        self,
        *args,
        name: Optional[str] = None,
        variables: Optional[Dict[str, str]] = None,
        template: Optional[str] = None,
        **kwargs,
    ) -> Union[WrappedMessage, WrappedAsyncStream[anthropic.types.MessageStreamEvent]]:
        start_timestamp = datetime.now(timezone.utc)
        stream_or_completion = await super().create(*args, **kwargs)
        first_token_timestamp = datetime.now(timezone.utc)

        messages: List[anthropic.types.MessageParam] = kwargs.pop("messages", [])
        wrapped_completion: Union[WrappedAsyncStream, WrappedMessage, None] = None

        if isinstance(stream_or_completion, anthropic.types.Message):
            output = stream_or_completion.content[0].text
            if not name:
                name = output[:20]

            wrapped_completion = WrappedMessage(
                client=self._client,
                trace_id=self._client.trace_id,
                name=name,
                template=template,
                start_timestamp=start_timestamp,
                end_timestamp=first_token_timestamp,
                first_token_timestamp=first_token_timestamp,
                completion_id=str(uuid4()),
                config_params=kwargs,
                input_messages=messages,
                **stream_or_completion.model_dump(),
            )
        else:
            if not name:
                name = "Anthropic Stream"

            wrapped_completion = WrappedAsyncStream(
                response=stream_or_completion.response,
                client=self._client,
                trace_id=self._client.trace_id,
                name=name,
                template=template,
                start_timestamp=start_timestamp,
                end_timestamp=None,
                first_token_timestamp=first_token_timestamp,
                config_params=kwargs,
                input_messages=messages,
                completion_id=str(uuid4()),
            )

        if wrapped_completion is None:
            return stream_or_completion

        if variables:
            for key, value in variables.items():
                wrapped_completion.variable(key, value)
        else:
            wrapped_completion.submit_to_baserun()

        return wrapped_completion

    @copy_type_hints(anthropic.resources.AsyncMessages)
    def stream(
        self,
        *args,
        name: Optional[str] = None,
        variables: Optional[Dict[str, str]] = None,
        template: Optional[str] = None,
        **kwargs,
    ) -> WrappedAsyncMessageStreamManager[anthropic.types.MessageStreamEvent]:
        if not name:
            name = "Anthropic Stream"

        start_timestamp = datetime.now(timezone.utc)
        stream_manager: anthropic.AsyncMessageStreamManager[anthropic.AsyncMessageStream] = super().stream(
            *args, **kwargs
        )
        request = stream_manager._AsyncMessageStreamManager__api_request  # type: ignore[attr-defined]
        first_token_timestamp = datetime.now(timezone.utc)

        messages: List[anthropic.types.MessageParam] = kwargs.pop("messages", [])

        return WrappedAsyncMessageStreamManager(
            api_request=request,
            stream_params={
                "client": self._client,
                "trace_id": self._client.trace_id,
                "name": name,
                "template": template,
                "start_timestamp": start_timestamp,
                "end_timestamp": None,
                "first_token_timestamp": first_token_timestamp,
                "config_params": kwargs,
                "input_messages": messages,
                "completion_id": str(uuid4()),
            },
        )


class WrappedAnthropicBaseClient(ClientMixin):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    messages: Union[WrappedSyncMessages, WrappedAsyncMessages]
    integrations: List[Integration]

    def __init__(
        self,
        client: Union[anthropic.Anthropic, anthropic.AsyncAnthropic],
        *args,
        name: Optional[str] = None,
        user: Optional[str] = None,
        session: Optional[str] = None,
        trace_id: Optional[str] = None,
        api_client: Optional[ApiClient] = None,
        api_key: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.name = name or "Unnamed"
        self.tags: List[Tag] = []
        self.evals: List[TraceEval] = []
        self.client = client
        self.trace_id = trace_id or str(uuid4())
        self._output = None
        self.error: Union[str, None] = None
        self.user = user
        self.session = session
        self.start_timestamp = datetime.now(timezone.utc)
        self.end_timestamp: Union[datetime, None] = None
        self.api_client = api_client or ApiClient(api_key=api_key)
        self.metadata = metadata or {}
        self.request_ids: Dict[str, str] = {}
        self.integrations = []

        try:
            from baserun.integrations.llamaindex import LLamaIndexInstrumentation

            self.integrate(LLamaIndexInstrumentation)
        except ImportError:
            pass

    @property
    def output(self):
        return self._output

    @output.setter
    def output(self, value):
        self._output = value
        self.submit_to_baserun()

    def genericize(self):
        return GenericClient(
            name=self.name,
            user=self.user,
            session=self.session,
            trace_id=self.trace_id,
            start_timestamp=self.start_timestamp,
            end_timestamp=self.end_timestamp,
            tags=self.tags,
            evals=self.evals,
            metadata=self.metadata,
            api_client=self.api_client,
            integrations=[],
            error=self.error,
        )

    def submit_to_baserun(self):
        if self.api_client:
            self.api_client.submit_trace(self.genericize())


class WrappedSyncAnthropicClient(WrappedAnthropicBaseClient, anthropic.Anthropic):
    messages: WrappedSyncMessages

    @copy_type_hints(anthropic.Anthropic)
    def __init__(
        self,
        client: anthropic.Anthropic,
        *args,
        name: Optional[str] = None,
        session: Optional[str] = None,
        user: Optional[str] = None,
        trace_id: Optional[str] = None,
        **kwargs,
    ):
        api_client = kwargs.pop("api_client", None)
        api_key = kwargs.pop("api_key", None)
        WrappedAnthropicBaseClient.__init__(
            self,
            client,
            *args,
            **kwargs,
            name=name,
            user=user,
            session=session,
            trace_id=trace_id,
            api_client=api_client,
            api_key=api_key,
        )
        anthropic.Anthropic.__init__(self, *args, **kwargs)
        self.messages = WrappedSyncMessages(self)


class WrappedAsyncAnthropicClient(WrappedAnthropicBaseClient, anthropic.AsyncAnthropic):
    messages: WrappedAsyncMessages

    @copy_type_hints(anthropic.AsyncAnthropic)
    def __init__(
        self,
        client: anthropic.AsyncAnthropic,
        *args,
        name: Optional[str] = None,
        session: Optional[str] = None,
        user: Optional[str] = None,
        trace_id: Optional[str] = None,
        **kwargs,
    ):
        api_client = kwargs.pop("api_client", None)
        api_key = kwargs.pop("api_key", None)
        WrappedAnthropicBaseClient.__init__(
            self,
            client,
            *args,
            **kwargs,
            name=name,
            user=user,
            session=session,
            trace_id=trace_id,
            api_client=api_client,
            api_key=api_key,
        )
        anthropic.AsyncAnthropic.__init__(self, *args, **kwargs)
        self.messages = WrappedAsyncMessages(self)


class Anthropic(WrappedSyncAnthropicClient):
    def __init__(self, *args, **kwargs):
        client = anthropic.Anthropic(*args, **kwargs)
        super().__init__(*args, **kwargs, client=client)


class AsyncAnthropic(WrappedAsyncAnthropicClient):
    def __init__(self, *args, **kwargs):
        client = anthropic.AsyncAnthropic(*args, **kwargs)
        super().__init__(*args, **kwargs, client=client)
