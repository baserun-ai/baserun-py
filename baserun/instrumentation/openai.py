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

    WRAPPED_METHODS = [
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
            "function": ChatCompletion.create,
            "span_name": "openai.chat",
        },
        {
            "class": Completion,
            "function": ChatCompletion.acreate,
            "span_name": "openai.chat",
        },
    ]

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    @staticmethod
    def set_request_attributes(span: Span, kwargs: dict[str, Any]):
        span.set_attribute(SpanAttributes.LLM_VENDOR, OPENAI_VENDOR_NAME)
        span.set_attribute(SpanAttributes.OPENAI_API_BASE, openai.api_base)
        span.set_attribute(SpanAttributes.OPENAI_API_TYPE, openai.api_type)

        span.set_attribute(SpanAttributes.LLM_TEMPERATURE, kwargs.get("temperature", 1))
        span.set_attribute(SpanAttributes.LLM_TOP_P, kwargs.get("top_p", 1))
        span.set_attribute(
            SpanAttributes.LLM_FREQUENCY_PENALTY,
            kwargs.get("frequency_penalty", 0),
        )
        span.set_attribute(
            SpanAttributes.LLM_PRESENCE_PENALTY,
            kwargs.get("presence_penalty", 0),
        )

        if functions := kwargs.get("functions"):
            span.set_attribute(SpanAttributes.LLM_FUNCTIONS, json.dumps(functions))

        if function_call := kwargs.get("function_call"):
            span.set_attribute(
                SpanAttributes.LLM_FUNCTION_CALL,
                json.dumps(function_call),
            )

        span.set_attribute(SpanAttributes.LLM_N, kwargs.get("n", 1))
        span.set_attribute(SpanAttributes.LLM_STREAM, kwargs.get("stream", False))

        if stop := kwargs.get("stop"):
            span.set_attribute(SpanAttributes.LLM_STOP, str(stop))

        if logit_bias := kwargs.get("logit_bias"):
            span.set_attribute(SpanAttributes.LLM_LOGIT_BIAS, json.dumps(logit_bias))

        if user := kwargs.get("user"):
            span.set_attribute(SpanAttributes.LLM_USER, user)

        messages = kwargs.get("messages", [])
        for i, message in enumerate(messages):
            prefix = f"{SpanAttributes.LLM_PROMPTS}.{i}"
            span.set_attribute(f"{prefix}.role", message.get("role"))
            span.set_attribute(f"{prefix}.content", message.get("content"))

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
                span.set_attribute(f"{prefix}.content", message.get("content"))
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
