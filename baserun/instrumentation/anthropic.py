#!/usr/bin/env python
import logging
from types import GeneratorType
from typing import Collection, Any

from anthropic.resources import Completions, AsyncCompletions
from anthropic.types import Completion
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import (
    unwrap,
)
from opentelemetry.sdk.trace import Span
from wrapt import wrap_function_wrapper

from baserun.instrumentation.span_attributes import (
    SpanAttributes,
    ANTHROPIC_VENDOR_NAME,
)
from baserun.instrumentation.wrappers import instrumented_wrapper

logger = logging.getLogger(__name__)

_instruments = ("anthropic >= 0.3.0",)
__version__ = "0.1.0"

WRAPPED_METHODS = [Completions.create, AsyncCompletions.create]


class AnthropicInstrumentor(BaseInstrumentor):
    """An instrumentor for OpenAI's client library."""

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def set_request_attributes(self, span: Span, kwargs: dict[str, Any]):
        span.set_attribute(SpanAttributes.LLM_VENDOR, ANTHROPIC_VENDOR_NAME)
        prefix = f"{SpanAttributes.LLM_PROMPTS}.0"
        span.set_attribute(f"{prefix}.content", kwargs.get("prompt"))

    def set_response_attributes(self, span: Span, response: Completion):
        prefix = f"{SpanAttributes.LLM_COMPLETIONS}.0"
        span.set_attribute(f"{prefix}.finish_reason", response.stop_reason)
        span.set_attribute(f"{prefix}.content", response.completion)
        span.set_attribute(SpanAttributes.ANTHROPIC_LOG_ID, response.log_id)

    def _instrument(self, **kwargs):
        return
        for wrapped_method in WRAPPED_METHODS:
            wrap_class = wrapped_method.get("class")
            wrap_method = wrapped_method.get("method")
            wrap_function_wrapper(
                "anthropic.resources.completions",
                f"{wrap_class}.{wrap_method}",
                instrumented_wrapper(self),
            )

    def _uninstrument(self, **kwargs):
        for wrapped_method in WRAPPED_METHODS:
            wrap_class = wrapped_method.get("class")
            unwrap(
                f"anthropic.resources.completions.{wrap_class}",
                wrapped_method.get("method"),
            )

    @staticmethod
    def generator_wrapper(original_generator: GeneratorType, span: Span):
        raise NotImplementedError
