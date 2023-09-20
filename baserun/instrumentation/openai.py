#!/usr/bin/env python
import json
import logging
from typing import Collection, Any

import openai
from openai.openai_object import OpenAIObject
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import (
    unwrap,
)
from opentelemetry.sdk.trace import Span
from opentelemetry.trace import get_tracer
from wrapt import wrap_function_wrapper

from baserun.instrumentation.span_attributes import SpanAttributes, OPENAI_VENDOR_NAME
from baserun.instrumentation.wrappers import instrumented_wrapper

logger = logging.getLogger(__name__)

_instruments = ("openai >= 0.27.0",)
__version__ = "0.1.0"

WRAPPED_METHODS = [
    {
        "object": "ChatCompletion",
        "method": "create",
        "span_name": "openai.chat",
    },
    {
        "object": "Completion",
        "method": "create",
        "span_name": "openai.completion",
    },
]


class OpenAIInstrumentor(BaseInstrumentor):
    """An instrumentor for OpenAI's client library."""

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def set_request_attributes(self, span: Span, kwargs: dict[str, Any]):
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

    def set_response_attributes(self, span: Span, response: OpenAIObject):
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

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)

        for wrapped_method in WRAPPED_METHODS:
            wrap_object = wrapped_method.get("object")
            wrap_method = wrapped_method.get("method")
            wrap_function_wrapper(
                "openai",
                f"{wrap_object}.{wrap_method}",
                instrumented_wrapper(
                    tracer=tracer, to_wrap=wrapped_method, instrumentor=self
                ),
            )

    def _uninstrument(self, **kwargs):
        for wrapped_method in WRAPPED_METHODS:
            wrap_object = wrapped_method.get("object")
            unwrap(f"openai.{wrap_object}", wrapped_method.get("method"))
