import json
import logging
from datetime import datetime, timezone
from functools import cached_property
from typing import Any, AsyncIterator, Coroutine, Dict, Iterator, List, Optional, Union, cast
from uuid import uuid4

import httpx
from openai import APIStatusError, AsyncStream
from openai import AsyncOpenAI as BaseAsyncOpenAI
from openai import OpenAI as BaseOpenAI
from openai._streaming import Stream, extract_stream_chunk_type
from openai._types import ResponseT
from openai.resources.chat import AsyncChat, AsyncCompletions, Chat, Completions
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessageParam
from openai.types.chat.chat_completion_chunk import Choice
from openai.types.completion_usage import CompletionUsage
from pydantic import BaseModel, ConfigDict, Field

from baserun.api import ApiClient
from baserun.integrations.integration import Integration
from baserun.mixins import ClientMixin, CompletionMixin
from baserun.models.evals import CompletionEval, TraceEval
from baserun.models.experiment import Experiment
from baserun.models.tags import Tag
from baserun.utils import copy_type_hints, count_prompt_tokens, deep_merge
from baserun.wrappers.generic import GenericChoice, GenericClient, GenericCompletion, GenericUsage

logger = logging.getLogger(__name__)


class WrappedChatCompletion(ChatCompletion, CompletionMixin):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    client: "WrappedOpenAIBaseClient"
    name: str
    error: Optional[str] = None
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

    def genericize(self):
        choices = [GenericChoice(**c.model_dump()) for c in self.choices]

        usage = GenericUsage(
            completion_tokens=self.usage.completion_tokens,
            prompt_tokens=self.usage.prompt_tokens,
            total_tokens=self.usage.total_tokens,
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
            input_messages=self.input_messages,
        )

    def submit_to_baserun(self):
        self.client.api_client.submit_completion(self.genericize())


class WrappedStreamBase(CompletionMixin, BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    client: "WrappedOpenAIBaseClient"
    id: Optional[str] = None
    name: str
    error: Optional[str] = None
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
        kwargs.pop("error", None)
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

    def genericize(self):
        merged_choice = deep_merge([c.model_dump() for c in self.captured_choices])
        merged_choice.pop("delta", None)
        merged_delta = deep_merge([c.delta.model_dump() for c in self.captured_choices])
        merged_choice["message"] = merged_delta or {"content": ""}

        usage = GenericUsage(
            completion_tokens=len(self.captured_choices) + 1,
            prompt_tokens=count_prompt_tokens(self.input_messages),
            total_tokens=count_prompt_tokens(self.input_messages) + len(self.captured_choices) + 1,
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
            client=self._client.genericize(),
            choices=[merged_choice],
            usage=usage,
            request_id=self.client.request_ids.get(self.id),
            tool_results=self.tool_results,
            tags=self.tags,
            evals=self.evals,
            input_messages=self.input_messages,
            config_params=self.config_params,
        )

    def submit_to_baserun(self):
        if not self.id:
            return

        if not self.end_timestamp:
            self.end_timestamp = datetime.now(timezone.utc)
        self.client.api_client.submit_stream(self.genericize())


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
            # Assemble deltas here instead of in api
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
        messages: List[ChatCompletionMessageParam] = kwargs.pop("messages", [])
        if not name:
            for message in reversed(messages):
                if message.get("content") and message.get("role") == "assistant":
                    name = str(message["content"])[:20]
                    break
        if not name:
            name = "OpenAI Completion"

        try:
            start_timestamp = datetime.now(timezone.utc)
            stream_or_completion = super().create(*args, **kwargs, messages=messages)
            first_token_timestamp = datetime.now(timezone.utc)
            wrapped: Union[WrappedChatCompletion, WrappedSyncStream, None] = None

            if isinstance(stream_or_completion, ChatCompletion):
                wrapped = WrappedChatCompletion(
                    **stream_or_completion.model_dump(),
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
        except APIStatusError as e:
            self._client.error = json.dumps(
                {
                    "exception": f"{e.__module__}.{e.__class__.__name__}",
                    "code": e.status_code,
                    "body": e.body,
                    "request_id": e.request_id,
                }
            )
            completion_id = str(uuid4())
            wrapped = WrappedChatCompletion(
                id=f"chatcompl-error-{completion_id}",
                client=self._client,
                trace_id=self._client.trace_id,
                name=name,
                template=template,
                start_timestamp=start_timestamp,
                end_timestamp=start_timestamp,
                first_token_timestamp=start_timestamp,
                config_params=kwargs,
                completion_id=completion_id,
                input_messages=messages,
                error=self._client.error,
                object="chat.completion",
                choices=[],
                created=int(datetime.now().timestamp()),
                usage=CompletionUsage(total_tokens=0, completion_tokens=0, prompt_tokens=0),
                **kwargs,
            )
            wrapped.submit_to_baserun()
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
    integrations: List[Integration]
    metadata: Dict[str, Any]
    autosubmit: bool = Field(default=True)

    def __init__(
        self,
        client: Union[BaseOpenAI, BaseAsyncOpenAI],
        *args,
        name: Optional[str] = None,
        user: Optional[str] = None,
        session: Optional[str] = None,
        trace_id: Optional[str] = None,
        api_client: Optional[ApiClient] = None,
        baserun_api_key: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        experiment: Optional[Experiment] = None,
    ):
        self.tags: List[Tag] = []
        self.evals: List[TraceEval] = []
        self.client = client
        self.trace_id = trace_id or str(uuid4())
        self.output = ""
        self.error: Union[str, None] = None
        self.user = user
        self.session = session
        self.start_timestamp = datetime.now(timezone.utc)
        self.end_timestamp: Union[datetime, None] = None
        self.metadata = metadata or {}
        self.request_ids: Dict[str, str] = {}
        self.integrations = []
        self.experiment = experiment

        if name:
            self.name = name
        else:
            self.name = ".".join([v for v in [client.organization, client.__class__.__name__] if v])

        self.api_client = api_client
        if not self.api_client:
            self.api_client = ApiClient(api_key=baserun_api_key)

        try:
            from baserun.integrations.llamaindex import LLamaIndexInstrumentation

            self.integrate(LLamaIndexInstrumentation)
        except ImportError:
            pass

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
            output=self.output,
            error=self.error,
            experiment=self.experiment,
        )

    def submit_to_baserun(self):
        if self.api_client:
            self.api_client.submit_trace(self.genericize())


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
        api_client: Optional[ApiClient] = None,
        baserun_api_key: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        experiment: Optional[Experiment] = None,
        **kwargs,
    ):
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
            baserun_api_key=baserun_api_key,
            metadata=metadata or {},
            experiment=experiment,
        )
        self.metadata = self.metadata
        self.name = self.name
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
        api_client: Optional[ApiClient] = None,
        baserun_api_key: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        experiment: Optional[Experiment] = None,
        **kwargs,
    ):
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
            baserun_api_key=baserun_api_key,
            metadata=metadata or {},
            experiment=experiment,
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
    def __init__(
        self,
        *args,
        name: Optional[str] = None,
        experiment: Optional[Experiment] = None,
        **kwargs,
    ):
        client = BaseOpenAI(*args, **kwargs)
        super().__init__(*args, **kwargs, client=client, name=name, experiment=experiment)


class AsyncOpenAI(WrappedAsyncOpenAIClient):
    def __init__(
        self,
        *args,
        name: Optional[str] = None,
        experiment: Optional[Experiment] = None,
        **kwargs,
    ):
        client = BaseAsyncOpenAI(*args, **kwargs)
        super().__init__(*args, **kwargs, client=client, name=name, experiment=experiment)
