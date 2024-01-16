import collections.abc
import logging
from typing import TYPE_CHECKING, Collection

from opentelemetry.sdk.trace import _Span

from baserun.instrumentation.base_instrumentor import BaseInstrumentor

logger = logging.getLogger(__name__)

_instruments = ("openai >= 0.27.0",)
__version__ = "0.1.0"

if TYPE_CHECKING:
    pass


class OpenAIInstrumentor(BaseInstrumentor):
    """An instrumentor for OpenAI's client library."""

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    @staticmethod
    def generator_wrapper(original_generator: collections.abc.Iterator, span: _Span):
        for value in original_generator:
            OpenAIInstrumentor._handle_generator_value(value, span)

            yield value

    @staticmethod
    async def async_generator_wrapper(original_generator: collections.abc.AsyncIterator, span: _Span):
        logged_failure = False
        async for value in original_generator:
            try:
                OpenAIInstrumentor._handle_generator_value(value, span)
            except Exception as e:
                # Avoid spamming failures for each token
                if not logged_failure:
                    logger.info(f"Baserun couldn't handle generator value {value}: {e}")
                    logged_failure = True

            yield value

    @staticmethod
    def _handle_generator_value(value: "OpenAIObject", span: _Span):
        # Currently we only support one choice in streaming responses
        choice = value.choices[0]

        if hasattr(choice, "delta"):
            # Chat
            delta = choice.delta
            role = delta.role
            new_content = delta.content
            new_function_call: "OpenAIObject" = delta.function_call
        else:
            # Completion
            role = None
            new_content = choice.text
            new_function_call = None

        prefix = f"0"
        content_attribute = f"{prefix}.content"
        function_name_attribute = f"{prefix}.function_name"
        function_arguments_attribute = f"{prefix}.function_arguments"

        if role:
            span.set_attribute(f"{prefix}.role", role)

        if new_content:
            content = span.attributes.get(content_attribute, "")
            span.set_attribute(content_attribute, content + new_content)

        if new_function_call:
            function_name = span.attributes.get(function_name_attribute, "")
            if name_delta := new_function_call.name:
                span.set_attribute(function_name_attribute, function_name + name_delta)

            function_arguments = span.attributes.get(function_arguments_attribute, "")
            if arguments_delta := new_function_call.arguments:
                span.set_attribute(function_arguments_attribute, function_arguments + arguments_delta)

        if ((new_content is None and not new_function_call) or choice.finish_reason) and span.is_recording():
            span.end()
