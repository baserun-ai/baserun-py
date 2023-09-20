#!/usr/bin/env python
import logging
from typing import Collection, Any

from openai import Completion
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import (
    unwrap,
)
from opentelemetry.sdk.trace import Span
from opentelemetry.trace import get_tracer
from wrapt import wrap_function_wrapper

from baserun.instrumentation.span_attributes import SpanAttributes
from baserun.instrumentation.wrappers import instrumented_wrapper

logger = logging.getLogger(__name__)

_instruments = ("anthropic >= 0.3.0",)
__version__ = "0.1.0"

WRAPPED_METHODS = [
    {
        "object": "Completions",
        "method": "create",
        "span_name": "anthropic.completions.create",
    },
    {
        "object": "AsyncCompletions",
        "method": "create",
        "span_name": "async_completions.create",
    },
]


class AnthropicInstrumentor(BaseInstrumentor):
    """An instrumentor for OpenAI's client library."""

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def set_request_attributes(self, span: Span, kwargs: dict[str, Any]):
        span.set_attribute(SpanAttributes.LLM_VENDOR, "Anthropic")
        prefix = f"{SpanAttributes.LLM_PROMPTS}.0"
        span.set_attribute(f"{prefix}.content", kwargs.get("prompt"))

    def set_response_attributes(self, span: Span, response: Completion):
        prefix = f"{SpanAttributes.LLM_COMPLETIONS}.0"
        span.set_attribute(f"{prefix}.finish_reason", response.stop_reason)
        span.set_attribute(f"{prefix}.content", response.completion)
        span.set_attribute(SpanAttributes.ANTHROPIC_LOG_ID, response.log_id)

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)
        for wrapped_method in WRAPPED_METHODS:
            wrap_object = wrapped_method.get("object")
            wrap_method = wrapped_method.get("method")
            wrap_function_wrapper(
                "anthropic.resources.completions",
                f"{wrap_object}.{wrap_method}",
                instrumented_wrapper(
                    tracer=tracer, to_wrap=wrapped_method, instrumentor=self
                ),
            )

    def _uninstrument(self, **kwargs):
        for wrapped_method in WRAPPED_METHODS:
            wrap_object = wrapped_method.get("object")
            unwrap(
                f"anthropic.resources.completions.{wrap_object}",
                wrapped_method.get("method"),
            )
