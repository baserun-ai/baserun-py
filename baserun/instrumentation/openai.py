import json
import logging
from types import GeneratorType
from typing import Collection, Any

import openai
from openai import ChatCompletion, Completion
from openai.openai_object import OpenAIObject
from opentelemetry.sdk.trace import Span

from baserun.instrumentation.base_instrumentor import BaseInstrumentor
from baserun.instrumentation.span_attributes import SpanAttributes, OPENAI_VENDOR_NAME

logger = logging.getLogger(__name__)

_instruments = ("openai >= 0.27.0",)
__version__ = "0.1.0"


class OpenAIInstrumentor(BaseInstrumentor):
    """An instrumentor for OpenAI's client library."""

    @staticmethod
    def wrapped_methods() -> list[dict[str, Any]]:
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
    def set_request_attributes(span: Span, kwargs: dict[str, Any]):
        span.set_attribute(SpanAttributes.LLM_VENDOR, OPENAI_VENDOR_NAME)
        span.set_attribute(SpanAttributes.OPENAI_API_BASE, openai.api_base)
        span.set_attribute(SpanAttributes.OPENAI_API_TYPE, openai.api_type)

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

        if (prompt := kwargs.get("prompt")) and not messages:
            span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.0.content", prompt)

    @staticmethod
    def set_response_attributes(span: Span, response: OpenAIObject):
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
                    span.set_attribute(
                        f"{prefix}.function_call",
                        json.dumps(message.get("function_call").to_dict()),
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
    def generator_wrapper(original_generator: GeneratorType, span: Span):
        for value in original_generator:
            # Currently we only support one choice in streaming responses
            prefix = f"{SpanAttributes.LLM_COMPLETIONS}.0"

            role = value.choices[0].delta.get("role")
            if role:
                span.set_attribute(f"{prefix}.role", role)

            new_content = value.choices[0].delta.get("content")
            if new_content:
                span.set_attribute(
                    f"{prefix}.content",
                    span.attributes.get(f"{prefix}.content", "") + new_content,
                )

            if new_content is None or value.choices[0].finish_reason:
                span.end()

            yield value
