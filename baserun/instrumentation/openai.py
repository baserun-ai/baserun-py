#!/usr/bin/env python
import logging
from types import GeneratorType
from typing import Collection

import openai
from opentelemetry import context as context_api, trace
from opentelemetry.context import get_value
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import (
    _SUPPRESS_INSTRUMENTATION_KEY,
    unwrap,
)
from opentelemetry.sdk.trace import Span
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


@with_tracer_wrapper
def instrumented_wrapper(tracer, to_wrap, wrapped, instance, args, kwargs):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    name = to_wrap.get("span_name")

    # Start new span
    span = tracer.start_span(
        name,
        kind=SpanKind.CLIENT,
        attributes={
            SpanAttributes.LLM_VENDOR: "OpenAI",
            SpanAttributes.LLM_REQUEST_TYPE: name.split(".")[-1],
        },
    )

    # Activate the span in the current context, but don't end it automatically
    with trace.use_span(span, end_on_exit=False):
        span.set_attribute(
            SpanAttributes.BASERUN_RUN_ID, get_value(SpanAttributes.BASERUN_RUN_ID)
        )

        # Capture request attributes
        # noinspection PyBroadException
        try:
            span.set_attribute(SpanAttributes.OPENAI_API_BASE, openai.api_base)
            span.set_attribute(SpanAttributes.OPENAI_API_TYPE, openai.api_type)
            span.set_attribute(SpanAttributes.LLM_REQUEST_MODEL, kwargs.get("model"))
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
                span.set_attribute(SpanAttributes.LLM_REQUEST_MAX_TOKENS, max_tokens)

            messages = kwargs.get("messages", [])
            for i, message in enumerate(messages):
                prefix = f"{SpanAttributes.LLM_PROMPTS}.{i}"
                span.set_attribute(f"{prefix}.role", message.get("role"))
                span.set_attribute(f"{prefix}.content", message.get("content"))

        except Exception as e:
            logger.warning(
                f"Failed to set input attributes for openai span, error: {e}"
            )

        # Actually call the wrapped method
        response = wrapped(*args, **kwargs)

        # If this is a streaming response, wrap it so we can capture each chunk
        if isinstance(response, GeneratorType):
            wrapped_response = generator_wrapper(response, span)
            return wrapped_response

        # If it's a full response capture the response attributes
        if response:
            # noinspection PyBroadException
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
            except Exception as e:
                logger.warning(
                    f"Failed to set response attributes for openai span, error: {e}"
                )
        else:
            # Not sure when this could happen?
            span.set_status(
                Status(description="No response received", status_code=StatusCode.UNSET)
            )

    # End the span and return the response
    span.end()
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
