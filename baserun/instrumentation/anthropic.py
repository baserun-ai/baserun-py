#!/usr/bin/env python
import collections.abc
import json
import logging
from typing import Collection, Any

from opentelemetry.sdk.trace import _Span

from baserun import Baserun
from baserun.instrumentation.base_instrumentor import BaseInstrumentor
from baserun.instrumentation.span_attributes import (
    SpanAttributes,
    ANTHROPIC_VENDOR_NAME,
)
from baserun.templates import most_similar_templates, best_guess_template_parameters

logger = logging.getLogger(__name__)

_instruments = ("anthropic >= 0.3.0",)
__version__ = "0.1.0"


class AnthropicInstrumentor(BaseInstrumentor):
    """An instrumentor for OpenAI's client library."""

    @staticmethod
    def wrapped_methods() -> list[dict[str, Any]]:
        from anthropic.resources import Completions, AsyncCompletions

        return [
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
    def set_request_attributes(span: _Span, kwargs: dict[str, Any]):
        span.set_attribute(SpanAttributes.LLM_VENDOR, ANTHROPIC_VENDOR_NAME)
        span.set_attribute(SpanAttributes.LLM_REQUEST_MODEL, kwargs.get("model"))

        if "top_k" in kwargs:
            span.set_attribute(SpanAttributes.LLM_TOP_K, kwargs.get("top_k"))

        if "top_p" in kwargs:
            span.set_attribute(SpanAttributes.LLM_TOP_P, kwargs.get("top_p"))

        if "stream" in kwargs:
            span.set_attribute(SpanAttributes.LLM_STREAM, kwargs.get("stream"))

        if "temperature" in kwargs:
            span.set_attribute(
                SpanAttributes.LLM_TEMPERATURE, kwargs.get("temperature")
            )

        if stop_sequences := kwargs.get("stop_sequences"):
            span.set_attribute(SpanAttributes.LLM_CHAT_STOP_SEQUENCES, stop_sequences)

        span.set_attribute(
            SpanAttributes.LLM_REQUEST_MAX_TOKENS, kwargs.get("max_tokens_to_sample")
        )

        prompt = kwargs.get("prompt")
        prefix = f"{SpanAttributes.LLM_PROMPTS}.0"
        span.set_attribute(f"{prefix}.content", prompt)

        templates_by_similarity = most_similar_templates(prompt)
        if templates_by_similarity and templates_by_similarity[0].template:
            most_similar_version = templates_by_similarity[0]
            if most_similar_version.id not in Baserun.used_template_parameters:
                most_similar_version = most_similar_version.template.active_version

            matched_parameters = best_guess_template_parameters(
                template_version=most_similar_version, prompt=prompt
            )

            span.set_attribute(
                SpanAttributes.BASERUN_TEMPLATE_VERSION_ID, most_similar_version.id
            )
            span.set_attribute(
                SpanAttributes.BASERUN_TEMPLATE_PARAMETERS,
                json.dumps(matched_parameters),
            )
            span.set_attribute(
                SpanAttributes.BASERUN_TEMPLATE_STRING,
                most_similar_version.template_string,
            )

    @staticmethod
    def set_response_attributes(span: _Span, response):
        prefix = f"{SpanAttributes.LLM_COMPLETIONS}.0"
        span.set_attribute(f"{prefix}.finish_reason", response.stop_reason)
        span.set_attribute(f"{prefix}.content", response.completion)
        span.set_attribute(SpanAttributes.ANTHROPIC_LOG_ID, response.log_id)

    @staticmethod
    def generator_wrapper(original_generator: collections.abc.Iterator, span: _Span):
        for value in original_generator:
            AnthropicInstrumentor._handle_generator_value(value, span)

            yield value

    @staticmethod
    async def async_generator_wrapper(
        original_generator: collections.abc.AsyncIterator, span: _Span
    ):
        async for value in original_generator:
            AnthropicInstrumentor._handle_generator_value(value, span)

            yield value

    @staticmethod
    def _handle_generator_value(value, span: _Span):
        new_content = value.completion

        prefix = f"{SpanAttributes.LLM_COMPLETIONS}.0"
        content_attribute = f"{prefix}.content"

        if new_content:
            content = span.attributes.get(content_attribute, "")
            span.set_attribute(content_attribute, content + new_content)

        if new_content is None or value.stop_reason:
            span.end()
