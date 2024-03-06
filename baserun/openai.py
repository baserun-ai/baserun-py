import json
from datetime import datetime
from typing import List, TYPE_CHECKING
from uuid import UUID

import openai
from httpx import URL

from baserun.grpc import get_or_create_submission_service
from baserun.v1.baserun_pb2 import Message, ToolFunction, ToolCall, Span, Run, SubmitSpanRequest, Status

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion
    from openai.types.chat.chat_completion import Choice


def compile_tool_calls(choice: "Choice") -> list["ToolCall"]:
    calls: List[ToolCall] = []
    if not choice.message.tool_calls:
        return calls

    for call in choice.message.tool_calls:
        calls.append(
            ToolCall(
                id=call.id,
                type=call.type,
                function=ToolFunction(name=call.function.name, arguments=call.function.arguments),
            )
        )

    return calls


def get_messages_from_completion(completion: "ChatCompletion") -> List[Message]:
    from openai.types.chat.chat_completion_message import FunctionCall

    completion_messages = []
    for choice in completion.choices:
        raw_function_call = choice.message.function_call
        if isinstance(raw_function_call, FunctionCall):
            function_call = raw_function_call.model_dump_json()
        elif isinstance(raw_function_call, str):
            function_call = raw_function_call
        else:
            function_call = json.dumps(raw_function_call)

        message = Message(
            role=choice.message.role,
            content=choice.message.content,
            finish_reason=choice.finish_reason,
            function_call=function_call,
            tool_calls=compile_tool_calls(choice),
            system_fingerprint=completion.system_fingerprint,
        )
        completion_messages.append(message)

    return completion_messages


def get_span_from_completion(completion: "ChatCompletion", trace: Run, **kwargs) -> Span:
    completion_messages = get_messages_from_completion(completion)

    prompt_messages = []
    for message in kwargs.get("messages", []):
        proto_message = Message(role=message.get("role"), content=message.get("content"))
        if message.get("tool_calls"):
            proto_message.tool_calls = message.get("tool_calls")

        prompt_messages.append(proto_message)

    span = Span(
        # Assuming that by here there's no error possible
        status=Status(message="ok", code=200),
        run_id=trace.run_id,
        trace_id=UUID(trace.run_id).bytes,
        prompt_messages=prompt_messages,
        completions=completion_messages,
        model=completion.model,
        total_tokens=completion.usage.total_tokens,
        completion_tokens=completion.usage.completion_tokens,
        prompt_tokens=completion.usage.prompt_tokens,
        completion_id=completion.id,
        name=kwargs.get("name", "openai.chat"),
        vendor=kwargs.get("vendor", "openai"),
    )
    span.start_time.FromSeconds(completion.created)
    span.end_time.FromDatetime(datetime.utcnow())

    base_url = openai.base_url
    if isinstance(base_url, URL):
        span.api_base = str(base_url.raw)
    else:
        span.api_base = base_url or "https://api.openai.com/v1"

    set_span_values_from_kwargs(span, **kwargs)

    return span


def submit_openai_completion(completion: "ChatCompletion", trace: Run, **kwargs) -> Span:
    span = get_span_from_completion(completion, trace, **kwargs)
    submit_span(span, trace)
    return span


def submit_span(span: Span, run: Run):
    return get_or_create_submission_service().SubmitSpan(SubmitSpanRequest(span=span, run=run))


def set_span_values_from_kwargs(span: Span, **kwargs) -> Span:
    span.stream = kwargs.get("stream", False)

    if max_tokens := kwargs.get("max_tokens"):
        span.max_tokens = max_tokens
    if temperature := kwargs.get("temperature"):
        span.temperature = temperature
    if top_p := kwargs.get("top_p"):
        span.top_p = top_p
    if top_k := kwargs.get("top_k"):
        span.top_k = top_k
    if frequency_penalty := kwargs.get("frequency_penalty"):
        span.frequency_penalty = frequency_penalty
    if presence_penalty := kwargs.get("presence_penalty"):
        span.presence_penalty = presence_penalty
    if n := kwargs.get("n"):
        span.n = n
    if logit_bias := kwargs.get("logit_bias"):
        span.logit_bias = logit_bias
    if logprobs := kwargs.get("logprobs"):
        span.logprobs = logprobs
    if echo := kwargs.get("echo"):
        span.echo = echo
    if suffix := kwargs.get("suffix"):
        span.suffix = suffix
    if best_of := kwargs.get("best_of"):
        span.best_of = best_of
    if user := kwargs.get("user"):
        span.user = user
    if function_call := kwargs.get("function_call"):
        span.function_call = json.dumps(function_call)
    if functions := kwargs.get("functions"):
        span.functions = json.dumps(functions)
    if tool_choice := kwargs.get("tool_choice"):
        span.tool_choice = json.dumps(tool_choice)
    if tools := kwargs.get("tools"):
        span.tools = json.dumps(tools)

    return span
