import collections.abc
import json
import logging
from typing import Collection, Any, TYPE_CHECKING, Union

import openai
from openai import Stream
from openai.types import CompletionChoice
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion import ChatCompletion
from opentelemetry.sdk.trace import _Span

from baserun.helpers import get_session_id
from baserun.instrumentation.base_instrumentor import BaseInstrumentor
from baserun.instrumentation.span_attributes import SpanAttributes, OPENAI_VENDOR_NAME
from baserun.templates import most_similar_templates, best_guess_template_parameters

logger = logging.getLogger(__name__)

_instruments = ("openai >= 0.27.0",)
__version__ = "0.1.0"

if TYPE_CHECKING:
    pass


class OpenAIInstrumentor(BaseInstrumentor):
    """An instrumentor for OpenAI's client library."""

    @staticmethod
    def wrapped_methods() -> list[dict[str, Any]]:
        from openai.resources import Completions, AsyncCompletions
        from openai.resources.chat import (
            Completions as ChatCompletions,
            AsyncCompletions as AsyncChatCompletions,
        )

        return [
            {
                "class": ChatCompletions,
                "function": ChatCompletions.create,
                "span_name": "openai.chat",
            },
            {
                "class": AsyncChatCompletions,
                "function": AsyncChatCompletions.create,
                "span_name": "openai.chat",
            },
            {
                "class": Completions,
                "function": Completions.create,
                "span_name": "openai.completion",
            },
            {
                "class": AsyncCompletions,
                "function": AsyncCompletions.create,
                "span_name": "openai.completion",
            },
        ]

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    @staticmethod
    def set_request_attributes(span: _Span, kwargs: dict[str, Any]):
        session_id = get_session_id()
        if session_id:
            span.set_attribute(SpanAttributes.BASERUN_SESSION_ID, session_id)

        span.set_attribute(SpanAttributes.LLM_VENDOR, OPENAI_VENDOR_NAME)
        span.set_attribute(
            SpanAttributes.OPENAI_API_BASE,
            openai.base_url or "https://api.openai.com/v1",
        )
        span.set_attribute(SpanAttributes.OPENAI_API_TYPE, openai.api_type or "open_ai")
        span.set_attribute(SpanAttributes.LLM_REQUEST_MODEL, kwargs.get("model"))

        max_tokens = kwargs.get("max_tokens")
        if max_tokens is not None:
            span.set_attribute(SpanAttributes.LLM_REQUEST_MAX_TOKENS, max_tokens)

        if "temperature" in kwargs:
            span.set_attribute(
                SpanAttributes.LLM_TEMPERATURE, kwargs.get("temperature")
            )

        if "top_p" in kwargs:
            span.set_attribute(SpanAttributes.LLM_TOP_P, kwargs.get("top_p"))

        if "frequency_penalty" in kwargs:
            span.set_attribute(
                SpanAttributes.LLM_FREQUENCY_PENALTY,
                kwargs.get("frequency_penalty"),
            )

        if "presence_penalty" in kwargs:
            span.set_attribute(
                SpanAttributes.LLM_PRESENCE_PENALTY,
                kwargs.get("presence_penalty"),
            )

        if "functions" in kwargs:
            span.set_attribute(
                SpanAttributes.LLM_FUNCTIONS, json.dumps(kwargs.get("functions"))
            )

        if "function_call" in kwargs:
            span.set_attribute(
                SpanAttributes.LLM_FUNCTION_CALL,
                json.dumps(kwargs.get("function_call")),
            )

        if "tools" in kwargs:
            span.set_attribute(
                SpanAttributes.LLM_TOOLS, json.dumps(kwargs.get("tools"))
            )

        if "tool_choice" in kwargs:
            span.set_attribute(
                SpanAttributes.LLM_TOOL_CHOICE,
                json.dumps(kwargs.get("tool_choice")),
            )

        if "n" in kwargs:
            span.set_attribute(SpanAttributes.LLM_N, kwargs.get("n"))

        if "stream" in kwargs:
            span.set_attribute(SpanAttributes.LLM_STREAM, kwargs.get("stream"))

        if stop := kwargs.get("stop"):
            if isinstance(stop, str):
                span.set_attribute(SpanAttributes.LLM_CHAT_STOP_SEQUENCES, [stop])
            else:
                span.set_attribute(SpanAttributes.LLM_CHAT_STOP_SEQUENCES, stop)

        if "logit_bias" in kwargs:
            span.set_attribute(
                SpanAttributes.LLM_LOGIT_BIAS, json.dumps(kwargs.get("logit_bias"))
            )

        if "logprobs" in kwargs:
            span.set_attribute(SpanAttributes.LLM_LOGPROBS, kwargs.get("logprobs"))

        if "echo" in kwargs:
            span.set_attribute(SpanAttributes.LLM_ECHO, kwargs.get("echo"))

        if "suffix" in kwargs:
            span.set_attribute(SpanAttributes.LLM_SUFFIX, kwargs.get("suffix"))

        if "best_of" in kwargs:
            span.set_attribute(SpanAttributes.LLM_BEST_OF, kwargs.get("best_of"))

        if "user" in kwargs:
            span.set_attribute(SpanAttributes.LLM_USER, kwargs.get("user"))

        messages = kwargs.get("messages", [])
        template_version = None
        formatted_prompt = ""
        for i, message in enumerate(messages):
            prefix = f"{SpanAttributes.LLM_PROMPTS}.{i}"
            span.set_attribute(f"{prefix}.role", message.get("role"))

            if content := message.get("content"):
                formatted_prompt += content

                if not template_version:
                    templates_by_similarity = most_similar_templates(content)
                    if templates_by_similarity:
                        template_version = templates_by_similarity[0]

                span.set_attribute(f"{prefix}.content", content)

            if function_call := message.get("function_call"):
                span.set_attribute(f"{prefix}.function_call", json.dumps(function_call))

        if (prompt := kwargs.get("prompt")) and not messages:
            formatted_prompt = prompt
            if not template_version:
                templates_by_similarity = most_similar_templates(prompt)
                if templates_by_similarity:
                    template_version = templates_by_similarity[0]

            span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.0.content", prompt)

        if template_version:
            span.set_attribute(
                SpanAttributes.BASERUN_TEMPLATE_VERSION_ID,
                template_version.id,
            )
            span.set_attribute(
                SpanAttributes.BASERUN_TEMPLATE_STRING,
                template_version.template_string,
            )
            matched_parameters = best_guess_template_parameters(
                template_version=template_version, prompt=formatted_prompt
            )
            if matched_parameters:
                span.set_attribute(
                    SpanAttributes.BASERUN_TEMPLATE_PARAMETERS,
                    json.dumps(matched_parameters),
                )

    @staticmethod
    def set_response_attributes(
        span: _Span, response: Union[ChatCompletion, Stream[ChatCompletionChunk]]
    ):
        span.set_attribute(SpanAttributes.LLM_COMPLETION_ID, response.id)

        choices = response.choices
        for i, choice in enumerate(choices):
            prefix = f"{SpanAttributes.LLM_COMPLETIONS}.{i}"
            span.set_attribute(f"{prefix}.finish_reason", choice.finish_reason)

            if isinstance(choice, CompletionChoice):
                span.set_attribute(f"{prefix}.content", choice.text)
            else:
                message = choice.message
                span.set_attribute(f"{prefix}.role", message.role)

                if content := message.content:
                    span.set_attribute(f"{prefix}.content", content)

                if tool_calls := message.tool_calls:
                    for tool_index, tool_call in enumerate(tool_calls):
                        tool_prefix = f"{prefix}.tool_calls.{tool_index}"
                        span.set_attribute(f"{tool_prefix}.id", tool_call.id)
                        span.set_attribute(f"{tool_prefix}.type", tool_call.type)
                        span.set_attribute(
                            f"{tool_prefix}.name", tool_call.function.name
                        )
                        span.set_attribute(
                            f"{tool_prefix}.function_arguments",
                            tool_call.function.arguments,
                        )

                if function_call := message.function_call:
                    span.set_attribute(f"{prefix}.function_name", function_call.name)
                    span.set_attribute(
                        f"{prefix}.function_arguments", function_call.arguments
                    )

        usage = response.usage
        if usage:
            span.set_attribute(
                SpanAttributes.LLM_USAGE_TOTAL_TOKENS, usage.total_tokens
            )
            span.set_attribute(
                SpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
                usage.completion_tokens,
            )
            span.set_attribute(
                SpanAttributes.LLM_USAGE_PROMPT_TOKENS,
                usage.prompt_tokens,
            )

    @staticmethod
    def generator_wrapper(original_generator: collections.abc.Iterator, span: _Span):
        for value in original_generator:
            OpenAIInstrumentor._handle_generator_value(value, span)

            yield value

    @staticmethod
    async def async_generator_wrapper(
        original_generator: collections.abc.AsyncIterator, span: _Span
    ):
        async for value in original_generator:
            OpenAIInstrumentor._handle_generator_value(value, span)

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

        prefix = f"{SpanAttributes.LLM_COMPLETIONS}.0"
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
                span.set_attribute(
                    function_arguments_attribute, function_arguments + arguments_delta
                )

        if (
            (new_content is None and not new_function_call) or choice.finish_reason
        ) and span.is_recording():
            span.end()
