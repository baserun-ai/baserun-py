from datetime import datetime
from random import randint
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterable,
    AsyncIterator,
    Callable,
    Iterable,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Type,
    TypedDict,
    Union,
)
from uuid import UUID

import anthropic.resources as ar
import httpx
from anthropic import AsyncStream, MessageStream, MessageStreamManager, MessageStreamT, Stream
from anthropic._types import Body, Headers, NotGiven, Query
from anthropic.lib.streaming._messages import AsyncMessageStream, AsyncMessageStreamManager, AsyncMessageStreamT
from anthropic.types import (
    ContentBlock,
    ContentBlockDeltaEvent,
    ImageBlockParam,
    MessageDeltaEvent,
    MessageParam,
    MessageStartEvent,
    MessageStreamEvent,
    TextBlockParam,
    message_create_params,
)
from anthropic.types import Message as AnthropicMessage

from baserun.helpers import BaserunProvider, BaserunType
from baserun.instrumentation.instrumentation import Instrumentation
from baserun.templates_util import FormattedContentString
from baserun.v1.baserun_pb2 import Message, Run, Span, SubmitSpanRequest

if TYPE_CHECKING:
    from baserun.baserun import _Baserun


def update_on_event(span: Span, msg: Optional[Message], event: MessageStreamEvent) -> Tuple[Span, Optional[Message]]:
    if isinstance(event, MessageStartEvent):
        msg = Message(role=event.message.role)
        span.prompt_tokens += event.message.usage.input_tokens
    elif isinstance(event, ContentBlockDeltaEvent):
        if msg and event.delta.type == "text_delta":
            msg.content += event.delta.text
    elif isinstance(event, MessageDeltaEvent):
        span.completion_tokens += event.usage.output_tokens
        if msg:
            msg.finish_reason += event.delta.stop_reason
            span.completions.append(msg)
    return span, msg


class IteratorWrapper:
    def __init__(
        self,
        iterator: Stream[MessageStreamEvent],
        span: Span,
        submit_span: Callable[[Span], None],
    ) -> None:
        self.iterator = iterator
        self.span = span
        self.submit_span = submit_span

    def __iter__(self) -> Iterator[MessageStreamEvent]:
        msg: Optional[Message] = None
        # there's accumulate_event that MessageStream uses, but it's not counting tokens ://
        for event in self.iterator:
            self.span, msg = update_on_event(self.span, msg, event)
            yield event
        self.submit_span(self.span)


class AsyncIteratorWrapper:
    def __init__(
        self,
        iterator: AsyncStream[MessageStreamEvent],
        span: Span,
        submit_span: Callable[[Span], None],
    ) -> None:
        self.iterator = iterator
        self.span = span
        self.submit_span = submit_span

    async def __aiter__(self) -> AsyncIterator[MessageStreamEvent]:
        msg: Optional[Message] = None
        async for event in self.iterator:
            self.span, msg = update_on_event(self.span, msg, event)
            yield event
        self.submit_span(self.span)


class OriginalMethod(NamedTuple):
    patched_obj: Any
    method: Callable


class CommonKwargs(TypedDict, total=False):
    max_tokens: int  # would do Required[int] but it's not supported in python<3.11
    messages: Iterable[MessageParam]  # Required
    model: str  # Required
    metadata: Union[message_create_params.Metadata, NotGiven]
    stop_sequences: Union[List[str], NotGiven]
    system: Union[str, NotGiven]
    temperature: Union[float, NotGiven]
    top_k: Union[int, NotGiven]
    top_p: Union[float, NotGiven]
    extra_headers: Optional[Headers]
    extra_query: Optional[Query]
    extra_body: Optional[Body]
    timeout: Optional[Union[float, httpx.Timeout, NotGiven]]


class CreateKwargs(CommonKwargs):
    stream: Union[bool, NotGiven]


class StreamKwargs(CommonKwargs):
    event_handler: Type[MessageStreamT]


def get_on_end_method(
    span: Span, process_response: Callable[[Span, Message], Span], submit_span: Callable[[Span], None]
):
    on_end_already_called = False

    def on_end(self: MessageStreamT) -> None:
        nonlocal on_end_already_called, span
        super(self.__class__, self).on_end()
        # on_end is called twice. probably unintended but that's what it is, so we need to handle that
        if not on_end_already_called:
            # TODO: correctly count tokens as anthropic.lib.streaming._messages.accumulate_event neglects it
            if msg := self._MessageStream__final_message_snapshot:
                span = process_response(span, msg)
            submit_span(span)
            on_end_already_called = True

    return on_end


def get_on_end_method_async(
    span: Span, process_response: Callable[[Span, Message], Span], submit_span: Callable[[Span], None]
):
    on_end_already_called = False

    async def on_end(self: AsyncMessageStreamT) -> None:
        nonlocal on_end_already_called, span
        await super(self.__class__, self).on_end()
        if not on_end_already_called:
            if msg := self._AsyncMessageStream__final_message_snapshot:
                span = process_response(span, msg)
            submit_span(span)
            on_end_already_called = True

    return on_end


class AnthropicInstrumentation(Instrumentation):
    def __init__(self, baserun_inst: "_Baserun") -> None:
        super().__init__(baserun_inst)
        self.original_methods: List[OriginalMethod] = []

    def instrument(self) -> None:
        self.patch_create()
        self.patch_stream()
        self.patch_create_async()
        self.patch_stream_async()

    def uninstrument(self) -> None:
        while self.original_methods:
            method = self.original_methods.pop()
            setattr(method.patched_obj, method.method.__name__, method.method)

    def process_kwargs(self, kwargs: CommonKwargs) -> Span:
        # TODO: what do we do with system message?

        span = Span(
            request_type=BaserunType.CHAT,
            name=f"{BaserunProvider.ANTHROPIC}.{BaserunType.CHAT}",
            vendor=BaserunProvider.ANTHROPIC,
            prompt_messages=self.anthropic_message_params_to_messages(kwargs["messages"]),
            model=kwargs["model"],
            max_tokens=kwargs["max_tokens"],
            # not sure if that's what Span.stop is for
            # mypy does some weird stuff here with typeddict...
            stop=None if isinstance(v := kwargs.get("stop_sequences"), NotGiven) else v,  # type: ignore
            top_p=None if isinstance(v := kwargs.get("top_p"), NotGiven) else v,
            top_k=None if isinstance(v := kwargs.get("top_k"), NotGiven) else v,
            temperature=None if isinstance(v := kwargs.get("temperature"), NotGiven) else v,
        )

        for msg in kwargs["messages"]:
            if isinstance(s := msg["content"], FormattedContentString):
                span.template_id = s.template_data.template_id
                for key, value in s.template_data.variables.items():
                    self.baserun.submit_input_variable(key, value, template_id=s.template_data.template_id)
                break

        span.start_time.FromDatetime(datetime.utcnow())
        return span

    def process_response_message(self, span: Span, msg: AnthropicMessage) -> Span:
        span.prompt_tokens = msg.usage.input_tokens
        span.completion_tokens = msg.usage.output_tokens
        span.total_tokens = msg.usage.input_tokens + msg.usage.output_tokens
        span.completions.extend(
            [
                Message(
                    role=msg.role,
                    finish_reason=msg.stop_reason,
                    content=self.anthropic_message_blocks_to_text(msg.content),
                )
            ]
        )
        return span

    def submit_span(self, span: Span) -> None:
        span.end_time.FromDatetime(datetime.utcnow())
        current_run: Run = self.baserun.get_or_create_current_run()
        span.run_id = current_run.run_id
        span.total_tokens = span.prompt_tokens + span.completion_tokens
        # not sure how should one set these trace_id and span_id. I don't really like it byt I'll set it like in
        #  the openai instrumentation. js sdk sets these to zeros...so maybe it doesn't matter anyway
        span.trace_id = UUID(current_run.run_id).bytes
        span.span_id = randint(0, 9999999999999999)
        span_request = SubmitSpanRequest(span=span, run=current_run)
        self.baserun.exporter_queue.put(span_request)

    def patch_create(self) -> None:
        func = ar.Messages.create
        self.original_methods.append(OriginalMethod(ar.Messages, func))

        def wrapper(
            obj: ar.Messages,
            **kwargs_,
        ) -> Union[Message, Iterable[MessageStreamEvent]]:
            # the best thing I could think of since typing.Unpack is unavailable in this python version
            kwargs: CreateKwargs = kwargs_  # type: ignore[assignment]
            span = self.process_kwargs(kwargs)
            span.stream = False if isinstance(v := kwargs.get("stream"), NotGiven) else v or False
            ret = func(obj, **kwargs)
            if kwargs.get("stream") is True:
                return IteratorWrapper(ret, span, self.submit_span)
            span = self.process_response_message(span, ret)
            self.submit_span(span)
            return ret

        ar.Messages.create = wrapper

    def patch_stream(self) -> None:
        func = ar.Messages.stream
        self.original_methods.append(OriginalMethod(ar.Messages, func))

        def wrapper(
            obj: ar.Messages,
            **kwargs_,
        ) -> MessageStreamManager[MessageStreamT]:
            kwargs: StreamKwargs = kwargs_  # type: ignore[assignment]
            span = self.process_kwargs(kwargs)
            span.stream = True
            orig_event_handler = kwargs.get("event_handler", MessageStream)
            new_event_handler = type(
                "BaserunMessageStream",
                (orig_event_handler,),
                {"on_end": get_on_end_method(span, self.process_response_message, self.submit_span)},
            )
            kwargs["event_handler"] = new_event_handler
            return func(obj, **kwargs)

        ar.Messages.stream = wrapper

    def patch_create_async(self) -> None:
        func = ar.AsyncMessages.create
        self.original_methods.append(OriginalMethod(ar.AsyncMessages, func))

        async def wrapper(
            obj: ar.AsyncMessages,
            **kwargs_,
        ) -> Union[Message, AsyncIterable[MessageStreamEvent]]:
            kwargs: CreateKwargs = kwargs_  # type: ignore[assignment]
            span = self.process_kwargs(kwargs)
            span.stream = False if isinstance(v := kwargs.get("stream"), NotGiven) else v or False
            ret = await func(obj, **kwargs)
            if kwargs.get("stream") is True:
                return AsyncIteratorWrapper(ret, span, self.submit_span)
            span = self.process_response_message(span, ret)
            self.submit_span(span)
            return ret

        ar.AsyncMessages.create = wrapper

    def patch_stream_async(self) -> None:
        func = ar.AsyncMessages.stream
        self.original_methods.append(OriginalMethod(ar.AsyncMessages, func))

        def wrapper(
            obj: ar.AsyncMessages,
            **kwargs_,
        ) -> AsyncMessageStreamManager[AsyncMessageStreamT]:
            kwargs: StreamKwargs = kwargs_  # type: ignore[assignment]
            span = self.process_kwargs(kwargs)
            span.stream = True
            orig_event_handler = kwargs.get("event_handler", AsyncMessageStream)
            new_event_handler = type(
                "BaserunAsyncMessageStream",
                (orig_event_handler,),
                {"on_end": get_on_end_method_async(span, self.process_response_message, self.submit_span)},
            )
            kwargs["event_handler"] = new_event_handler
            return func(obj, **kwargs)

        ar.AsyncMessages.stream = wrapper

    @staticmethod
    def anthropic_message_blocks_to_text(blocks: Iterable[Union[TextBlockParam, ImageBlockParam, ContentBlock]]) -> str:
        text = ""
        for part in blocks:
            if isinstance(part, dict) and part["type"] == "text":
                text += part["text"]
            elif isinstance(part, ContentBlock):
                text += part.text
            # ignoring image parts
        return text

    def anthropic_message_params_to_messages(self, messages: Iterable[MessageParam]) -> List[Message]:
        ret = []
        for msg in messages:
            if isinstance(msg["content"], str):
                ret.append(Message(role=msg["role"], content=msg["content"]))
                continue

            ret.append(Message(role=msg["role"], content=self.anthropic_message_blocks_to_text(msg["content"])))

        return ret
