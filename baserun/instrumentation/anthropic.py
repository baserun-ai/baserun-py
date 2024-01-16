#!/usr/bin/env python
import collections.abc
import logging
from typing import Collection

from opentelemetry.sdk.trace import _Span

from baserun.instrumentation.base_instrumentor import BaseInstrumentor

logger = logging.getLogger(__name__)

_instruments = ("anthropic >= 0.3.0",)
__version__ = "0.1.0"


class AnthropicInstrumentor(BaseInstrumentor):
    """An instrumentor for Anthropic's client library."""

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    @staticmethod
    def generator_wrapper(original_generator: collections.abc.Iterator, span: _Span):
        for value in original_generator:
            AnthropicInstrumentor._handle_generator_value(value, span)

            yield value

    @staticmethod
    async def async_generator_wrapper(original_generator: collections.abc.AsyncIterator, span: _Span):
        logged_failure = False
        async for value in original_generator:
            try:
                AnthropicInstrumentor._handle_generator_value(value, span)
            except Exception as e:
                # Avoid spamming failures for each token
                if not logged_failure:
                    logger.info(f"Baserun couldn't handle generator value {value}: {e}")
                    logged_failure = True

            yield value

    @staticmethod
    def _handle_generator_value(value, span: _Span):
        new_content = value.completion

        prefix = f"0"
        content_attribute = f"{prefix}.content"

        if new_content:
            content = span.attributes.get(content_attribute, "")
            span.set_attribute(content_attribute, content + new_content)

        if new_content is None or value.stop_reason and span.is_recording():
            span.end()
