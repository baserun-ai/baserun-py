#!/usr/bin/env python
import logging
from typing import Collection

import openai
from opentelemetry import context as context_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import (
    _SUPPRESS_INSTRUMENTATION_KEY,
    unwrap,
)
from opentelemetry.trace import get_tracer, SpanKind
from opentelemetry.trace.status import Status, StatusCode
from wrapt import wrap_function_wrapper

from baserun.instrumentation.span_attributes import SpanAttributes

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


def with_tracer_wrapper(func):
    def with_tracer(tracer, to_wrap):
        def wrapper(wrapped, instance, args, kwargs):
            # prevent double wrapping
            if hasattr(wrapped, "__wrapped__"):
                return wrapped(*args, **kwargs)

            return func(tracer, to_wrap, wrapped, instance, args, kwargs)

        return wrapper

    return with_tracer


@with_tracer_wrapper
def instrumented_wrapper(tracer, to_wrap, wrapped, instance, args, kwargs):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    name = to_wrap.get("span_name")
    with tracer.start_as_current_span(
        name,
        kind=SpanKind.CLIENT,
        attributes={
            SpanAttributes.LLM_VENDOR: "OpenAI",
            SpanAttributes.LLM_REQUEST_TYPE: name.split(".")[-1],
        },
    ) as span:
        # Set request attributes
        if span.is_recording():
            try:
                span.set_attribute(SpanAttributes.OPENAI_API_BASE, openai.api_base)
                span.set_attribute(SpanAttributes.OPENAI_API_TYPE, openai.api_type)
                span.set_attribute(
                    SpanAttributes.LLM_REQUEST_MODEL, kwargs.get("model")
                )
                span.set_attribute(
                    SpanAttributes.LLM_TEMPERATURE, kwargs.get("temperature", 1)
                )
                span.set_attribute(SpanAttributes.LLM_TOP_P, kwargs.get("top_p", 1))
                span.set_attribute(
                    SpanAttributes.LLM_FREQUENCY_PENALTY,
                    kwargs.get("frequency_penalty", 0),
                )
                span.set_attribute(
                    SpanAttributes.LLM_PRESENCE_PENALTY,
                    kwargs.get("presence_penalty", 0),
                )

                max_tokens = kwargs.get("max_tokens")
                if max_tokens:
                    span.set_attribute(
                        SpanAttributes.LLM_REQUEST_MAX_TOKENS, max_tokens
                    )

                messages = kwargs.get("messages", [])
                for i, message in enumerate(messages):
                    prefix = f"{SpanAttributes.LLM_PROMPTS}.{i}"
                    span.set_attribute(f"{prefix}.role", message.get("role"))
                    span.set_attribute(f"{prefix}.content", message.get("content"))

            except Exception as ex:  # pylint: disable=broad-except
                logger.warning(
                    "Failed to set input attributes for openai span, error: %s", str(ex)
                )

        # Actually call the wrapped method

        response = wrapped(*args, **kwargs)
        # Set response attributes
        if response and span.is_recording():
            try:
                span.set_status(Status(StatusCode.OK))
                choices = response.get("choices", [])
                for i, choice in enumerate(choices):
                    prefix = f"{SpanAttributes.LLM_COMPLETIONS}.{i}"
                    span.set_attribute(
                        f"{prefix}.finish_reason", choice.get("finish_reason")
                    )

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

            except Exception as ex:  # pylint: disable=broad-except
                logger.warning(
                    "Failed to set response attributes for openai span, error: %s",
                    str(ex),
                )

        return response


class OpenAIInstrumentor(BaseInstrumentor):
    """An instrumentor for OpenAI's client library."""

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)
        for wrapped_method in WRAPPED_METHODS:
            wrap_object = wrapped_method.get("object")
            wrap_method = wrapped_method.get("method")
            wrap_function_wrapper(
                "openai",
                f"{wrap_object}.{wrap_method}",
                instrumented_wrapper(tracer, wrapped_method),
            )

    def _uninstrument(self, **kwargs):
        for wrapped_method in WRAPPED_METHODS:
            wrap_object = wrapped_method.get("object")
            unwrap(f"openai.{wrap_object}", wrapped_method.get("method"))
