#!/usr/bin/env python
import logging
from types import GeneratorType
from typing import Collection, Any

from anthropic.resources import Completions, AsyncCompletions
from anthropic.types import Completion
from opentelemetry.sdk.trace import Span

from baserun.instrumentation.base_instrumentor import BaseInstrumentor
from baserun.instrumentation.span_attributes import (
    SpanAttributes,
    ANTHROPIC_VENDOR_NAME,
)

logger = logging.getLogger(__name__)

_instruments = ("anthropic >= 0.3.0",)
__version__ = "0.1.0"


class AnthropicInstrumentor(BaseInstrumentor):
    """An instrumentor for OpenAI's client library."""

    WRAPPED_METHODS = [
        {
            "class": Completions,
            "function": Completions.create,
            "span_name": "anthropic.completion",
        },
        {
            "class": AsyncCompletions,
            "function": AsyncCompletions.create,
            "span_name": "anthropic.completion",
        },
    ]

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    @staticmethod
    def set_request_attributes(span: Span, kwargs: dict[str, Any]):
        span.set_attribute(SpanAttributes.LLM_VENDOR, ANTHROPIC_VENDOR_NAME)
        span.set_attribute(SpanAttributes.LLM_REQUEST_MODEL, kwargs.get("model"))
        if stop_sequences := kwargs.get("stop_sequences"):
            span.set_attribute(SpanAttributes.LLM_CHAT_STOP_SEQUENCES, stop_sequences)

        span.set_attribute(
            SpanAttributes.LLM_REQUEST_MAX_TOKENS, kwargs.get("max_tokens_to_sample")
        )

        prefix = f"{SpanAttributes.LLM_PROMPTS}.0"
        span.set_attribute(f"{prefix}.content", kwargs.get("prompt"))

    @staticmethod
    def set_response_attributes(span: Span, response: Completion):
        prefix = f"{SpanAttributes.LLM_COMPLETIONS}.0"
        span.set_attribute(f"{prefix}.finish_reason", response.stop_reason)
        span.set_attribute(f"{prefix}.content", response.completion)
        span.set_attribute(SpanAttributes.ANTHROPIC_LOG_ID, response.log_id)

    @staticmethod
    def generator_wrapper(original_generator: GeneratorType, span: Span):
        raise NotImplementedError
