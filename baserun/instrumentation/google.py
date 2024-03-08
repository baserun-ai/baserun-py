from copy import deepcopy
from datetime import datetime
from random import randint
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterable,
    Callable,
    Iterable,
    List,
    MutableSequence,
    NamedTuple,
    Optional,
    Union,
)
from uuid import UUID

import google.ai.generativelanguage as glm

from baserun.helpers import BaserunProvider, BaserunType
from baserun.instrumentation.instrumentation import Instrumentation
from baserun.v1.baserun_pb2 import Message, Run, Span, SubmitSpanRequest

if TYPE_CHECKING:
    from baserun.baserun import _Baserun


def reduce_responses_func(
    acc: Optional[glm.GenerateContentResponse], val: glm.GenerateContentResponse
) -> glm.GenerateContentResponse:
    # we're merging it to a single response object in order to treat it as a single message in a span
    if not acc:
        # copy to make sure that modifying this later on does not affect anything we don't want it to
        return deepcopy(val)
    if not acc.candidates or not val.candidates:
        # not sure if that's actually possible at all
        return acc
    # for streaming there's always one candidate
    acc.candidates[0].finish_reason = val.candidates[0].finish_reason
    acc.candidates[0].content.parts.extend(val.candidates[0].content.parts)
    return acc


class AsyncIteratorWrapper:
    def __init__(
        self,
        iterator: AsyncIterable[glm.GenerateContentResponse],
        process_collected_response: Callable[[glm.GenerateContentResponse], None],
    ) -> None:
        self.iterator = iterator
        self.process_collected_response = process_collected_response

    async def __aiter__(self):
        collected_response = None
        async for ret in self.iterator:
            collected_response = reduce_responses_func(collected_response, ret)
            yield ret
        self.process_collected_response(collected_response)


class IteratorWrapper:
    def __init__(
        self,
        iterator: Iterable[glm.GenerateContentResponse],
        process_collected_response: Callable[[glm.GenerateContentResponse], None],
    ) -> None:
        self.iterator = iterator
        self.process_collected_response = process_collected_response

    def __iter__(self):
        collected_response = None
        for ret in self.iterator:
            collected_response = reduce_responses_func(collected_response, ret)
            yield ret
        self.process_collected_response(collected_response)


class OriginalMethod(NamedTuple):
    patched_obj: Any
    method: Callable


class GoogleInstrumentation(Instrumentation):
    def __init__(self, baserun_inst: "_Baserun") -> None:
        super().__init__(baserun_inst)
        self.original_methods: List[OriginalMethod] = []

    def instrument(self) -> None:
        self.patch_generate_content()
        self.patch_generate_content_async()
        self.patch_stream_generate_content()
        self.patch_stream_generate_content_async()

    def uninstrument(self) -> None:
        while self.original_methods:
            method = self.original_methods.pop()
            setattr(method.patched_obj, method.method.__name__, method.method)

    @staticmethod
    def get_final_request(
        request: Optional[Union[glm.GenerateContentRequest, dict]] = None,
        model: Optional[str] = None,
        contents: Optional[MutableSequence[glm.Content]] = None,
    ) -> glm.GenerateContentRequest:
        if not isinstance(request, glm.GenerateContentRequest):
            request = glm.GenerateContentRequest(request)
            if model is not None:
                request.model = model
            if contents is not None:
                request.contents = contents
        return request

    def process_request(self, request: glm.GenerateContentRequest) -> Span:
        # TODO: add function calls once google's python lib supports it (api itself has some support)
        span = Span(
            request_type=BaserunType.CHAT,
            name=f"{BaserunProvider.GOOGLE}.{BaserunType.CHAT}",
            vendor=BaserunProvider.GOOGLE,
            prompt_messages=self.google_contents_to_messages(request.contents),
            model=request.model,
            # not sure if that's what Span.stop is for
            stop=request.generation_config.stop_sequences,
            max_tokens=request.generation_config.max_output_tokens,
            top_p=request.generation_config.top_p,
            top_k=request.generation_config.top_k,
            temperature=request.generation_config.temperature,
        )
        span.start_time.FromDatetime(datetime.utcnow())
        return span

    def process_response(self, span: Span, response: Optional[glm.GenerateContentResponse]) -> Span:
        span.end_time.FromDatetime(datetime.utcnow())
        if response:
            span.completions.extend(self.google_candidates_to_messages(response.candidates))
        return span

    def submit_span(self, span: Span) -> None:
        current_run: Run = self.baserun.get_or_create_current_run()
        span.run_id = current_run.run_id
        # not sure how should one set these trace_id and span_id. I don't really like it byt I'll set it like in
        #  the openai instrumentation. js sdk sets these to zeros...so maybe it doesn't matter anyway
        span.trace_id = UUID(current_run.run_id).bytes
        span.span_id = randint(0, 9999999999999999)

        # TODO: add counting tokens as an opt-in. we're not doing it now bcaz it requires making additional requests
        span.completion_tokens = 0
        span.prompt_tokens = 0
        span.total_tokens = 0

        span_request = SubmitSpanRequest(span=span, run=current_run)
        self.baserun.exporter_queue.put(span_request)

    def patch_generate_content(self) -> None:
        func = glm.GenerativeServiceClient.generate_content
        self.original_methods.append(OriginalMethod(glm.GenerativeServiceClient, func))

        def wrapper(
            obj: glm.GenerativeServiceClient,
            request: Optional[Union[glm.GenerateContentRequest, dict]] = None,
            *,
            model: Optional[str] = None,
            contents: Optional[MutableSequence[glm.Content]] = None,
            **kwargs,
        ) -> glm.GenerateContentResponse:
            request = self.get_final_request(request, model, contents)
            span = self.process_request(request)
            span.stream = False
            ret = func(obj, request, model=model, contents=contents, **kwargs)
            span = self.process_response(span, ret)
            self.submit_span(span)
            return ret

        glm.GenerativeServiceClient.generate_content = wrapper

    def patch_generate_content_async(self) -> None:
        func = glm.GenerativeServiceAsyncClient.generate_content
        self.original_methods.append(OriginalMethod(glm.GenerativeServiceAsyncClient, func))

        async def wrapper(
            obj: glm.GenerativeServiceAsyncClient,
            request: Optional[Union[glm.GenerateContentRequest, dict]] = None,
            *,
            model: Optional[str] = None,
            contents: Optional[MutableSequence[glm.Content]] = None,
            **kwargs,
        ) -> glm.GenerateContentResponse:
            request = self.get_final_request(request, model, contents)
            span = self.process_request(request)
            span.stream = False
            ret = await func(obj, request, model=model, contents=contents, **kwargs)
            span = self.process_response(span, ret)
            self.submit_span(span)
            return ret

        glm.GenerativeServiceAsyncClient.generate_content = wrapper

    def patch_stream_generate_content(self) -> None:
        func = glm.GenerativeServiceClient.stream_generate_content
        self.original_methods.append(OriginalMethod(glm.GenerativeServiceClient, func))

        def wrapper(
            obj: glm.GenerativeServiceClient,
            request: Optional[Union[glm.GenerateContentRequest, dict]] = None,
            *,
            model: Optional[str] = None,
            contents: Optional[MutableSequence[glm.Content]] = None,
            **kwargs,
        ) -> Iterable[glm.GenerateContentResponse]:
            request = self.get_final_request(request, model, contents)
            span = self.process_request(request)
            span.stream = True

            def process_collected_response(collected_response: glm.GenerateContentResponse):
                nonlocal span
                span = self.process_response(span, collected_response)
                self.submit_span(span)

            iterator = func(obj, request, model=model, contents=contents, **kwargs)
            return IteratorWrapper(iterator, process_collected_response)

        glm.GenerativeServiceClient.stream_generate_content = wrapper

    def patch_stream_generate_content_async(self) -> None:
        func = glm.GenerativeServiceAsyncClient.stream_generate_content
        self.original_methods.append(OriginalMethod(glm.GenerativeServiceAsyncClient, func))

        async def wrapper(
            obj: glm.GenerativeServiceAsyncClient,
            request: Optional[Union[glm.GenerateContentRequest, dict]] = None,
            *,
            model: Optional[str] = None,
            contents: Optional[MutableSequence[glm.Content]] = None,
            **kwargs,
        ) -> AsyncIterable[glm.GenerateContentResponse]:
            request = self.get_final_request(request, model, contents)
            span = self.process_request(request)
            span.stream = True

            def process_collected_response(collected_response: glm.GenerateContentResponse):
                nonlocal span
                span = self.process_response(span, collected_response)
                self.submit_span(span)

            iterator = await func(obj, request, model=model, contents=contents, **kwargs)
            return AsyncIteratorWrapper(iterator, process_collected_response)

        glm.GenerativeServiceAsyncClient.stream_generate_content = wrapper

    @staticmethod
    def google_contents_to_messages(contents: Iterable[glm.Content]) -> List[Message]:
        ret = []
        for content in contents:
            text = "".join(part.text for part in content.parts)
            ret.append(Message(role=content.role, content=text))
        return ret

    @staticmethod
    def google_candidates_to_messages(candidates: Iterable[glm.Candidate]) -> List[Message]:
        ret = []
        for candidate in candidates:
            text = "".join(part.text for part in candidate.content.parts)
            ret.append(Message(role=candidate.content.role, content=text, finish_reason=candidate.finish_reason.name))
        return ret
