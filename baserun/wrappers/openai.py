import logging
import traceback
from datetime import datetime, timezone
from functools import cached_property
from typing import Any, AsyncIterator, Coroutine, Dict, Iterator, List, Optional, Union, cast
from uuid import uuid4

import httpx
from openai import AsyncOpenAI as BaseAsyncOpenAI
from openai import AsyncStream
from openai import OpenAI as BaseOpenAI
from openai._streaming import Stream, extract_stream_chunk_type
from openai._types import ResponseT
from openai.resources.chat import AsyncChat, AsyncCompletions, Chat, Completions
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessageParam
from openai.types.chat.chat_completion_chunk import Choice
from pydantic import BaseModel, ConfigDict, Field

from baserun.api import ApiClient
from baserun.mixins import ClientMixin, CompletionMixin
from baserun.models.evals import CompletionEval, TraceEval
from baserun.models.tags import Tag
from baserun.utils import copy_type_hints

logger = logging.getLogger(__name__)


class WrappedChatCompletion(ChatCompletion, CompletionMixin):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    client: "WrappedOpenAIBaseClient"
    name: str
    trace_id: str
    completion_id: str
    template: Optional[str]
    start_timestamp: datetime
    first_token_timestamp: datetime
    end_timestamp: datetime
    input_messages: List[ChatCompletionMessageParam] = Field(default_factory=list)
    config_params: Dict[str, Any] = Field(default_factory=dict)
    tool_results: List[Any] = Field(default_factory=list)
    tags: List[Tag] = Field(default_factory=list)
    evals: List[CompletionEval] = Field(default_factory=list)

    def submit_to_baserun(self):
        self.client.api_client.submit_completion(self)


class WrappedStreamBase(CompletionMixin, BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    client: "WrappedOpenAIBaseClient"
    id: Optional[str] = None
    name: str
    trace_id: str
    completion_id: str
    template: Optional[str] = None
    start_timestamp: datetime
    end_timestamp: Optional[datetime] = None
    first_token_timestamp: Optional[datetime] = None
    input_messages: List[ChatCompletionMessageParam] = Field(default_factory=list)
    config_params: Dict[str, Any] = Field(default_factory=dict)
    tool_results: List[Any] = Field(default_factory=list)
    tags: List[Tag] = Field(default_factory=list)
    evals: List[CompletionEval] = Field(default_factory=list)

    # Used for evals on streaming responses
    captured_choices: List[Choice] = Field(default_factory=list)

    def __init__(self, *args, **kwargs):
        CompletionMixin.__init__(self)
        BaseModel.__init__(self, **kwargs)
        # Get rid of special kwargs that we take but OpenAI doesn't
        kwargs.pop("name", None)
        kwargs.pop("completion_id", None)
        kwargs.pop("trace_id", None)
        kwargs.pop("template", None)
        kwargs.pop("config_params", None)
        kwargs.pop("start_timestamp", None)
        kwargs.pop("end_timestamp", None)
        kwargs.pop("first_token_timestamp", None)
        kwargs.pop("tool_results", None)
        kwargs.pop("tags", None)
        kwargs.pop("evals", None)
        kwargs.pop("input_messages", None)
        AsyncStream.__init__(
            self,
            cast_to=extract_stream_chunk_type(AsyncStream[ChatCompletionChunk]),
            *args,
            **kwargs,
        )

    def submit_to_baserun(self):
        if not self.end_timestamp:
            self.end_timestamp = datetime.now(timezone.utc)
        self._client.api_client.submit_stream(self)


class WrappedAsyncStream(WrappedStreamBase, AsyncStream[ChatCompletionChunk]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._iterator = self.__stream__()

    async def __stream__(self) -> AsyncIterator[ChatCompletionChunk]:
        stream = super().__stream__()
        async for item in stream:
            # TODO? Support n > 1
            self.captured_choices.append(item.choices[0])
            self.id = item.id
            yield item

        self.submit_to_baserun()


class WrappedSyncStream(WrappedStreamBase, Stream[ChatCompletionChunk]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._iterator = self.__stream__()

    def __iter__(self) -> Iterator[ChatCompletionChunk]:  # type: ignore[override]
        for item in self._iterator:
            yield item

    def __stream__(self) -> Iterator[ChatCompletionChunk]:
        stream = super().__stream__()
        for item in stream:
            # TODO? Support n > 1
            self.captured_choices.append(item.choices[0])
            self.id = item.id
            yield item

        self.submit_to_baserun()


class WrappedCompletions(Completions):
    _client: "WrappedSyncOpenAIClient"

    @copy_type_hints(Completions)
    def create(
        self,
        *args,
        name: Optional[str] = None,
        variables: Optional[Dict[str, str]] = None,
        template: Optional[str] = None,
        **kwargs,
    ) -> Union[WrappedChatCompletion, WrappedSyncStream]:
        try:
            messages: List[ChatCompletionMessageParam] = kwargs.pop("messages", [])
            if not name:
                for message in reversed(messages):
                    if message.get("content") and message.get("role") == "assistant":
                        name = str(message["content"])[:20]
                        break
            if not name:
                name = "OpenAI Completion"

            start_timestamp = datetime.now(timezone.utc)
            stream_or_completion = super().create(*args, **kwargs, messages=messages)
            first_token_timestamp = datetime.now(timezone.utc)
            wrapped: Union[WrappedChatCompletion, WrappedSyncStream, None] = None

            if isinstance(stream_or_completion, ChatCompletion):
                wrapped = WrappedChatCompletion(
                    client=self._client,
                    trace_id=self._client.trace_id,
                    name=name,
                    template=template,
                    start_timestamp=start_timestamp,
                    first_token_timestamp=first_token_timestamp,
                    end_timestamp=first_token_timestamp,
                    config_params=kwargs,
                    completion_id=str(uuid4()),
                    input_messages=messages,
                    **stream_or_completion.model_dump(),
                )

            elif isinstance(stream_or_completion, Stream):
                wrapped = WrappedSyncStream(
                    response=stream_or_completion.response,
                    client=self._client,
                    trace_id=self._client.trace_id,
                    name=name,
                    template=template,
                    start_timestamp=start_timestamp,
                    end_timestamp=None,
                    first_token_timestamp=first_token_timestamp,
                    completion_id=str(uuid4()),
                    input_messages=messages,
                    config_params=kwargs,
                )
            else:
                logger.debug(f"Unexpected response type: {type(stream_or_completion)}")
                return stream_or_completion

            if variables:
                for key, value in variables.items():
                    wrapped.variable(key, value)
            else:
                wrapped.submit_to_baserun()

            return wrapped
        except Exception as e:
            # TODO: Probably should assemble a Completion object in an error state and submit that
            self._client.error = traceback.format_exc()
            self._client.submit_to_baserun()
            raise e


class WrappedAsyncCompletions(AsyncCompletions):
    _client: "WrappedAsyncOpenAIClient"

    @copy_type_hints(AsyncCompletions)
    async def create(
        self,
        *args,
        name: Optional[str] = None,
        variables: Optional[Dict[str, str]] = None,
        template: Optional[str] = None,
        **kwargs,
    ) -> Union[WrappedAsyncStream, WrappedChatCompletion]:
        start_timestamp = datetime.now(timezone.utc)
        stream_or_completion = await super().create(*args, **kwargs)
        first_token_timestamp = datetime.now(timezone.utc)

        messages: List[ChatCompletionMessageParam] = kwargs.pop("messages", [])
        wrapped_completion: Union[WrappedAsyncStream, WrappedChatCompletion, None] = None

        if isinstance(stream_or_completion, ChatCompletion):
            if not name:
                name = (stream_or_completion.choices[0].message.content or "OpenAI Completion")[:20]

            wrapped_completion = WrappedChatCompletion(
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
                name = "OpenAI Completion"

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

        if variables:
            for key, value in variables.items():
                wrapped_completion.variable(key, value)
        else:
            wrapped_completion.submit_to_baserun()

        return wrapped_completion


class WrappedChat(Chat):
    @cached_property
    def completions(self) -> WrappedCompletions:
        return WrappedCompletions(self._client)


class WrappedAsyncChat(AsyncChat):
    @cached_property
    def completions(self) -> WrappedAsyncCompletions:
        return WrappedAsyncCompletions(self._client)


class WrappedOpenAIBaseClient(ClientMixin):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    chat: WrappedChat | WrappedAsyncChat

    def __init__(
        self,
        client: Union[BaseOpenAI, BaseAsyncOpenAI],
        *args,
        name: Optional[str] = None,
        user: Optional[str] = None,
        session: Optional[str] = None,
        trace_id: Optional[str] = None,
        api_client: Optional[ApiClient] = None,
        api_key: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.name = name or ".".join([v for v in [client.organization, client.__class__.__name__] if v])
        self.tags: List[Tag] = []
        self.evals: List[TraceEval] = []
        self.client = client
        self.trace_id = trace_id or str(uuid4())
        self._result = None
        self.error: Union[str, None] = None
        self.user = user
        self.session = session
        self.start_timestamp = datetime.now(timezone.utc)
        self.end_timestamp: Union[datetime, None] = None
        self.api_client = api_client or ApiClient(self, api_key=api_key)
        self.metadata = metadata or {}
        self.request_ids: Dict[str, str] = {}

    @property
    def result(self):
        return self._result

    @result.setter
    def result(self, value):
        self._result = value
        self.submit_to_baserun()

    def submit_to_baserun(self):
        if self.api_client:
            self.api_client.submit_trace()


class WrappedSyncOpenAIClient(WrappedOpenAIBaseClient, BaseOpenAI):
    chat: WrappedChat

    @copy_type_hints(WrappedOpenAIBaseClient)
    def __init__(
        self,
        client: BaseAsyncOpenAI,
        *args,
        name: Optional[str] = None,
        session: Optional[str] = None,
        user: Optional[str] = None,
        trace_id: Optional[str] = None,
        **kwargs,
    ):
        api_client = kwargs.pop("api_client", None)
        api_key = kwargs.pop("api_key", None)
        WrappedOpenAIBaseClient.__init__(
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
        BaseOpenAI.__init__(self, *args, **kwargs)
        self.chat = WrappedChat(self)

    def _process_response(self, *, response: httpx.Response, cast_to: type[ResponseT], **kwargs) -> ResponseT:
        request_id = response.headers.get("x-request-id")
        processed = super()._process_response(response=response, cast_to=cast_to, **kwargs)
        if response.stream:
            # Streaming responses are handled with _process_response_data
            return processed

        completion_id = response.json().get("id")
        self.request_ids[completion_id] = request_id
        return processed

    def _process_response_data(
        self, *, response: httpx.Response, cast_to: type[ResponseT], data: object, **kwargs
    ) -> ResponseT:
        request_id = response.headers.get("x-request-id")
        data_dict: Dict[str, str] = cast(Dict[str, str], data)
        completion_id = data_dict.get("completion_id")
        if completion_id:
            self.request_ids[completion_id] = request_id

        return super()._process_response_data(response=response, data=data, cast_to=cast_to, **kwargs)


class WrappedAsyncOpenAIClient(WrappedOpenAIBaseClient, BaseAsyncOpenAI):
    chat: WrappedAsyncChat

    @copy_type_hints(WrappedOpenAIBaseClient)
    def __init__(
        self,
        client: BaseAsyncOpenAI,
        *args,
        name: Optional[str] = None,
        session: Optional[str] = None,
        user: Optional[str] = None,
        trace_id: Optional[str] = None,
        **kwargs,
    ):
        api_client = kwargs.pop("api_client", None)
        api_key = kwargs.pop("api_key", None)
        WrappedOpenAIBaseClient.__init__(
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
        BaseAsyncOpenAI.__init__(self, *args, **kwargs)
        self.chat = WrappedAsyncChat(self)

    def _process_response(
        self, *, response: httpx.Response, cast_to: type[ResponseT], **kwargs
    ) -> Coroutine[Any, Any, ResponseT]:
        request_id = response.headers.get("x-request-id")
        processed = super()._process_response(response=response, cast_to=cast_to, **kwargs)
        if response.stream:
            # Streaming responses are handled with _process_response_data
            return processed

        completion_id = response.json().get("id")
        self.request_ids[completion_id] = request_id
        return processed

    def _process_response_data(
        self, *, response: httpx.Response, cast_to: type[ResponseT], data: object, **kwargs
    ) -> ResponseT:
        request_id = response.headers.get("x-request-id")
        data_dict: Dict[str, str] = cast(Dict[str, str], data)
        completion_id = data_dict.get("completion_id")
        if completion_id:
            self.request_ids[completion_id] = request_id

        return super()._process_response_data(response=response, data=data, cast_to=cast_to, **kwargs)


class OpenAI(WrappedSyncOpenAIClient):
    def __init__(self, *args, **kwargs):
        client = BaseOpenAI(*args, **kwargs)
        super().__init__(*args, **kwargs, client=client)


class AsyncOpenAI(WrappedAsyncOpenAIClient):
    def __init__(self, *args, **kwargs):
        client = BaseAsyncOpenAI(*args, **kwargs)
        super().__init__(*args, **kwargs, client=client)
