import collections.abc
import json
import logging
from typing import Collection, Any, TYPE_CHECKING

import openai
from opentelemetry.sdk.trace import _Span

from baserun.instrumentation.base_instrumentor import BaseInstrumentor
from baserun.instrumentation.span_attributes import SpanAttributes, OPENAI_VENDOR_NAME

logger = logging.getLogger(__name__)

_instruments = ("openai >= 0.27.0",)
__version__ = "0.1.0"

if TYPE_CHECKING:
    from openai.openai_object import OpenAIObject


class OpenAIInstrumentor(BaseInstrumentor):
    """An instrumentor for OpenAI's client library."""

    @staticmethod
    def wrapped_methods() -> list[dict[str, Any]]:
        from openai import ChatCompletion, Completion

        return [
            {
                "class": ChatCompletion,
                "function": ChatCompletion.create,
                "span_name": "openai.chat",
            },
            {
                "class": ChatCompletion,
                "function": ChatCompletion.acreate,
                "span_name": "openai.chat",
            },
            {
                "class": Completion,
                "function": Completion.create,
                "span_name": "openai.completion",
            },
            {
                "class": Completion,
                "function": Completion.acreate,
                "span_name": "openai.completion",
            },
        ]

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    @staticmethod
    def set_request_attributes(span: _Span, kwargs: dict[str, Any]):
        span.set_attribute(SpanAttributes.LLM_VENDOR, OPENAI_VENDOR_NAME)
        span.set_attribute(SpanAttributes.OPENAI_API_BASE, openai.api_base)
        span.set_attribute(SpanAttributes.OPENAI_API_TYPE, openai.api_type)
        span.set_attribute(SpanAttributes.LLM_REQUEST_MODEL, kwargs.get("model"))

        max_tokens = kwargs.get("max_tokens")
        if max_tokens is not None:
            span.set_attribute(SpanAttributes.LLM_REQUEST_MAX_TOKENS, max_tokens)

        if "temperature" in kwargs:
            span.set_attribute(
                SpanAttributes.LLM_TEMPERATURE, kwargs.get("temperature")
            )

        if "top_p" in kwargs:
            span.set_attribute(SpanAttributes.LLM_TOP_P, kwargs.get("top_p"))

        if "frequency_penalty" in kwargs:
            span.set_attribute(
                SpanAttributes.LLM_FREQUENCY_PENALTY,
                kwargs.get("frequency_penalty"),
            )

        if "presence_penalty" in kwargs:
            span.set_attribute(
                SpanAttributes.LLM_PRESENCE_PENALTY,
                kwargs.get("presence_penalty"),
            )

        if "functions" in kwargs:
            span.set_attribute(
                SpanAttributes.LLM_FUNCTIONS, json.dumps(kwargs.get("functions"))
            )

        if "function_call" in kwargs:
            span.set_attribute(
                SpanAttributes.LLM_FUNCTION_CALL,
                json.dumps(kwargs.get("function_call")),
            )

        if "n" in kwargs:
            span.set_attribute(SpanAttributes.LLM_N, kwargs.get("n"))

        if "stream" in kwargs:
            span.set_attribute(SpanAttributes.LLM_STREAM, kwargs.get("stream"))

        if stop := kwargs.get("stop"):
            if isinstance(stop, str):
                span.set_attribute(SpanAttributes.LLM_CHAT_STOP_SEQUENCES, [stop])
            else:
                span.set_attribute(SpanAttributes.LLM_CHAT_STOP_SEQUENCES, stop)

        if "logit_bias" in kwargs:
            span.set_attribute(
                SpanAttributes.LLM_LOGIT_BIAS, json.dumps(kwargs.get("logit_bias"))
            )

        if "logprobs" in kwargs:
            span.set_attribute(SpanAttributes.LLM_LOGPROBS, kwargs.get("logprobs"))

        if "echo" in kwargs:
            span.set_attribute(SpanAttributes.LLM_ECHO, kwargs.get("echo"))

        if "suffix" in kwargs:
            span.set_attribute(SpanAttributes.LLM_SUFFIX, kwargs.get("suffix"))

        if "best_of" in kwargs:
            span.set_attribute(SpanAttributes.LLM_BEST_OF, kwargs.get("best_of"))

        if "user" in kwargs:
            span.set_attribute(SpanAttributes.LLM_USER, kwargs.get("user"))

        messages = kwargs.get("messages", [])
        for i, message in enumerate(messages):
            prefix = f"{SpanAttributes.LLM_PROMPTS}.{i}"
            span.set_attribute(f"{prefix}.role", message.get("role"))

            if content := message.get("content"):
                span.set_attribute(f"{prefix}.content", content)

            if function_call := message.get("function_call"):
                span.set_attribute(f"{prefix}.function_call", json.dumps(function_call))

        if (prompt := kwargs.get("prompt")) and not messages:
            span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.0.content", prompt)

    @staticmethod
    def set_response_attributes(span: _Span, response: "OpenAIObject"):
        choices = response.get("choices", [])
        for i, choice in enumerate(choices):
            prefix = f"{SpanAttributes.LLM_COMPLETIONS}.{i}"
            span.set_attribute(f"{prefix}.finish_reason", choice.get("finish_reason"))

            message = choice.get("message")
            text = choice.get("text")
            if message:
                span.set_attribute(f"{prefix}.role", message.get("role"))

                if content := message.get("content"):
                    span.set_attribute(f"{prefix}.content", content)

                if "function_call" in message:
                    function_call = message.get("function_call")
                    span.set_attribute(
                        f"{prefix}.function_name", function_call.get("name")
                    )
                    span.set_attribute(
                        f"{prefix}.function_arguments", function_call.get("arguments")
                    )

            elif text:
                span.set_attribute(f"{prefix}.content", text)

        usage = response.get("usage")
        if usage:
            span.set_attribute(
                SpanAttributes.LLM_USAGE_TOTAL_TOKENS, usage.get("total_tokens")
            )
            span.set_attribute(
                SpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
                usage.get("completion_tokens"),
            )
            span.set_attribute(
                SpanAttributes.LLM_USAGE_PROMPT_TOKENS,
                usage.get("prompt_tokens"),
            )

    @staticmethod
    def generator_wrapper(original_generator: collections.abc.Iterator, span: _Span):
        for value in original_generator:
            OpenAIInstrumentor._handle_generator_value(value, span)

            yield value

    @staticmethod
    async def async_generator_wrapper(
        original_generator: collections.abc.AsyncIterator, span: _Span
    ):
        async for value in original_generator:
            OpenAIInstrumentor._handle_generator_value(value, span)

            yield value

    @staticmethod
    def _handle_generator_value(value: "OpenAIObject", span: _Span):
        # Currently we only support one choice in streaming responses
        choice = value.choices[0]

        if hasattr(choice, "delta"):
            # Chat
            delta = choice.delta
            role = delta.get("role")
            new_content = delta.get("content")
            new_function_call: "OpenAIObject" = delta.get("function_call")
        else:
            # Completion
            role = None
            new_content = choice.text
            new_function_call = None

        prefix = f"{SpanAttributes.LLM_COMPLETIONS}.0"
        content_attribute = f"{prefix}.content"
        function_name_attribute = f"{prefix}.function_name"
        function_arguments_attribute = f"{prefix}.function_arguments"

        if role:
            span.set_attribute(f"{prefix}.role", role)

        if new_content:
            content = span.attributes.get(content_attribute, "")
            span.set_attribute(content_attribute, content + new_content)

        if new_function_call:
            function_name = span.attributes.get(function_name_attribute, "")
            if name_delta := new_function_call.get("name"):
                span.set_attribute(function_name_attribute, function_name + name_delta)

            function_arguments = span.attributes.get(function_arguments_attribute, "")
            if arguments_delta := new_function_call.get("arguments"):
                span.set_attribute(
                    function_arguments_attribute, function_arguments + arguments_delta
                )

        if (
            (new_content is None and not new_function_call) or choice.finish_reason
        ) and span.is_recording():
            span.end()
