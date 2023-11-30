import json
import logging
from time import sleep
from typing import Sequence, Any

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter

from baserun.constants import PARENT_SPAN_NAME
from baserun.instrumentation.span_attributes import (
    SpanAttributes,
    ANTHROPIC_VENDOR_NAME,
)
from baserun.v1.baserun_pb2 import (
    Status,
    Message,
    Span,
    SubmitSpanRequest,
    ToolCall,
    ToolFunction,
)
from .grpc import get_or_create_submission_service

logger = logging.getLogger(__name__)


class BaserunExporter(SpanExporter):
    @staticmethod
    def _extract_prefix_dicts(attributes, prefix) -> list[dict[str, Any]]:
        result = {}
        for key, value in attributes.items():
            if key.startswith(prefix):
                parts = key[len(prefix) + 1 :].split(".")
                index, field = int(parts[0]), parts[1]
                if index not in result:
                    result[index] = {}
                result[index][field] = value

        return [v for k, v in sorted(result.items())]

    def export(self, spans: Sequence[ReadableSpan]):
        from baserun import Baserun

        for span in spans:
            if span.name.startswith(PARENT_SPAN_NAME) or not span.name.startswith(
                "baserun"
            ):
                continue

            status = Status(
                message=span.status.description, code=span.status.status_code.value
            )
            prompt_messages = [
                Message(**message_attrs)
                for message_attrs in self._extract_prefix_dicts(
                    span.attributes, SpanAttributes.LLM_PROMPTS
                )
            ]

            completions = []
            for i, message_attrs in enumerate(
                self._extract_prefix_dicts(
                    span.attributes, SpanAttributes.LLM_COMPLETIONS
                )
            ):
                if "function_name" in message_attrs:
                    message_attrs["function_call"] = json.dumps(
                        {
                            "name": message_attrs.pop("function_name"),
                            "arguments": message_attrs.pop("function_arguments"),
                        }
                    )

                message_attrs.pop("tool_calls", "")
                message = Message(**message_attrs)

                tool_calls_prefix = f"{SpanAttributes.LLM_COMPLETIONS}.{i}.tool_calls"
                for tool_attrs in self._extract_prefix_dicts(
                    span.attributes, tool_calls_prefix
                ):
                    tool_name = tool_attrs.pop("name")
                    tool_args = tool_attrs.pop("function_arguments")
                    tool_attrs["function"] = ToolFunction(
                        name=tool_name, arguments=tool_args
                    )
                    tool_call = ToolCall(**tool_attrs)
                    message.tool_calls.append(tool_call)

                completions.append(message)

            # Trace IDs are huge integers, so they must be encoded as bytes
            trace_id_int = span.context.trace_id
            trace_id = trace_id_int.to_bytes(
                (trace_id_int.bit_length() + 7) // 8, "big"
            )

            vendor = span.attributes.get(SpanAttributes.LLM_VENDOR)

            run_str = span.attributes.get(SpanAttributes.BASERUN_RUN)
            if not run_str:
                logger.warning("Baserun run attribute not set, cannot submit run")
                return

            run = Baserun.deserialize_run(run_str)

            span_message = Span(
                run_id=run.run_id,
                trace_id=trace_id,
                span_id=span.context.span_id,
                name=span.name,
                vendor=vendor,
                status=status,
                total_tokens=span.attributes.get(
                    SpanAttributes.LLM_USAGE_TOTAL_TOKENS, 0
                ),
                completion_tokens=span.attributes.get(
                    SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, 0
                ),
                prompt_tokens=span.attributes.get(
                    SpanAttributes.LLM_USAGE_PROMPT_TOKENS, 0
                ),
                prompt_messages=prompt_messages,
                model=span.attributes.get(SpanAttributes.LLM_REQUEST_MODEL, ""),
                completions=completions,
                request_type=span.attributes.get(SpanAttributes.LLM_REQUEST_TYPE, ""),
                max_tokens=span.attributes.get(SpanAttributes.LLM_REQUEST_MAX_TOKENS),
                stop=span.attributes.get(SpanAttributes.LLM_CHAT_STOP_SEQUENCES),
                completion_id=span.attributes.get(SpanAttributes.LLM_COMPLETION_ID),
            )
            span_message.start_time.FromNanoseconds(span.start_time)
            span_message.end_time.FromNanoseconds(span.end_time)

            set_span_attr(
                span_message,
                "template_version_id",
                span,
                SpanAttributes.BASERUN_TEMPLATE_VERSION_ID,
            )
            set_span_attr(
                span_message,
                "template_string",
                span,
                SpanAttributes.BASERUN_TEMPLATE_STRING,
            )
            set_span_attr(
                span_message,
                "template_parameters",
                span,
                SpanAttributes.BASERUN_TEMPLATE_PARAMETERS,
            )

            set_span_attr(
                span_message, "temperature", span, SpanAttributes.LLM_TEMPERATURE
            )
            set_span_attr(span_message, "top_p", span, SpanAttributes.LLM_TOP_P)
            set_span_attr(span_message, "stream", span, SpanAttributes.LLM_STREAM)

            if vendor == ANTHROPIC_VENDOR_NAME:
                set_span_attr(span_message, "top_k", span, SpanAttributes.LLM_TOP_K)
                set_span_attr(
                    span_message, "log_id", span, SpanAttributes.ANTHROPIC_LOG_ID
                )
            else:
                set_span_attr(
                    span_message, "api_base", span, SpanAttributes.OPENAI_API_BASE
                )
                set_span_attr(
                    span_message, "api_base", span, SpanAttributes.OPENAI_API_BASE
                )
                set_span_attr(
                    span_message, "api_type", span, SpanAttributes.OPENAI_API_TYPE
                )
                set_span_attr(
                    span_message,
                    "response_format",
                    span,
                    SpanAttributes.LLM_RESPONSE_FORMAT,
                )
                set_span_attr(span_message, "seed", span, SpanAttributes.LLM_SEED)
                set_span_attr(span_message, "tools", span, SpanAttributes.LLM_TOOLS)
                set_span_attr(
                    span_message,
                    "tool_choice",
                    span,
                    SpanAttributes.LLM_TOOL_CHOICE,
                )
                set_span_attr(
                    span_message, "functions", span, SpanAttributes.LLM_FUNCTIONS
                )
                set_span_attr(
                    span_message,
                    "function_call",
                    span,
                    SpanAttributes.LLM_FUNCTION_CALL,
                )
                set_span_attr(span_message, "n", span, SpanAttributes.LLM_N)
                set_span_attr(
                    span_message,
                    "frequency_penalty",
                    span,
                    SpanAttributes.LLM_FREQUENCY_PENALTY,
                )
                set_span_attr(
                    span_message,
                    "presence_penalty",
                    span,
                    SpanAttributes.LLM_PRESENCE_PENALTY,
                )
                set_span_attr(
                    span_message, "logit_bias", span, SpanAttributes.LLM_LOGIT_BIAS
                )
                set_span_attr(
                    span_message, "logprobs", span, SpanAttributes.LLM_LOGPROBS
                )
                set_span_attr(span_message, "echo", span, SpanAttributes.LLM_ECHO)
                set_span_attr(span_message, "suffix", span, SpanAttributes.LLM_SUFFIX)
                set_span_attr(span_message, "best_of", span, SpanAttributes.LLM_BEST_OF)
                set_span_attr(span_message, "user", span, SpanAttributes.LLM_USER)

            span_request = SubmitSpanRequest(span=span_message, run=run)
            try:
                get_or_create_submission_service().SubmitSpan(span_request)
            except Exception as e:
                if hasattr(e, "details"):
                    # Race condition where the span is submitted before the run start call finishes
                    if "not found" in e.details():
                        sleep(5)
                        get_or_create_submission_service().SubmitSpan(span_request)
                    else:
                        logger.warning(
                            f"Failed to submit span to Baserun: {e.details()}"
                        )
                else:
                    logger.warning(f"Failed to submit span to Baserun: {e}")


def set_span_attr(
    span_message: Span,
    span_message_attribute: str,
    span: ReadableSpan,
    span_attribute: str,
):
    """Sets a value on the span message only if it's not None"""
    value = span.attributes.get(span_attribute)
    if value is not None:
        setattr(span_message, span_message_attribute, value)
