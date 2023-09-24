import logging
from typing import Sequence, Any

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter

from baserun.instrumentation.span_attributes import (
    SpanAttributes,
    ANTHROPIC_VENDOR_NAME,
)
from baserun.v1.baserun_pb2 import Status, Message, Span, SubmitSpanRequest

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
            if span.name.startswith("baserun.parent") or not span.name.startswith(
                "baserun"
            ):
                continue

            status = Status(
                message=span.status.description, code=span.status.status_code.value
            )
            prompt_messages = [
                Message(**message_attrs)
                for message_attrs in self._extract_prefix_dicts(
                    span.attributes, "llm.prompts"
                )
            ]
            completions = [
                Message(**message_attrs)
                for message_attrs in self._extract_prefix_dicts(
                    span.attributes, "llm.completions"
                )
            ]

            # Trace IDs are huge integers, so they must be encoded as bytes
            trace_id_int = span.context.trace_id
            trace_id = trace_id_int.to_bytes(
                (trace_id_int.bit_length() + 7) // 8, "big"
            )

            vendor = span.attributes.get(SpanAttributes.LLM_VENDOR)

            run_str = span.attributes.get(SpanAttributes.BASERUN_RUN)
            if not run_str:
                logger.warning("Baserun run attribute not set, cannot submit run")

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
            )
            span_message.start_time.FromNanoseconds(span.start_time)
            span_message.end_time.FromNanoseconds(span.end_time)

            if vendor == ANTHROPIC_VENDOR_NAME:
                span_message.log_id = span.attributes.get(
                    SpanAttributes.ANTHROPIC_LOG_ID, ""
                )
            else:
                span_message.request_type = span.attributes.get(
                    SpanAttributes.LLM_REQUEST_TYPE, ""
                )
                span_message.api_base = span.attributes.get(
                    SpanAttributes.OPENAI_API_BASE, ""
                )
                span_message.api_type = span.attributes.get(
                    SpanAttributes.OPENAI_API_TYPE, ""
                )

                if functions := span.attributes.get(SpanAttributes.LLM_FUNCTIONS):
                    span_message.functions = functions
                if function_call := span.attributes.get(
                    SpanAttributes.LLM_FUNCTION_CALL
                ):
                    span_message.function_call = function_call

                span_message.temperature = span.attributes.get(
                    SpanAttributes.LLM_TEMPERATURE, 1
                )
                span_message.top_p = span.attributes.get(SpanAttributes.LLM_TOP_P, 1)
                span_message.n = span.attributes.get(SpanAttributes.LLM_N, 1)
                span_message.stream = span.attributes.get(
                    SpanAttributes.LLM_STREAM, False
                )

                if stop := span.attributes.get(SpanAttributes.LLM_STOP):
                    span_message.stop = stop

                span_message.max_tokens = span.attributes.get(
                    SpanAttributes.LLM_REQUEST_MAX_TOKENS, 0
                )
                span_message.frequency_penalty = span.attributes.get(
                    SpanAttributes.LLM_FREQUENCY_PENALTY, 0
                )
                span_message.presence_penalty = span.attributes.get(
                    SpanAttributes.LLM_PRESENCE_PENALTY, 0
                )

                if logit_bias := span.attributes.get(SpanAttributes.LLM_LOGIT_BIAS):
                    span_message.logit_bias = logit_bias

                if user := span.attributes.get(SpanAttributes.LLM_USER):
                    span_message.user = user

            span_request = SubmitSpanRequest(span=span_message, run=run)
            try:
                Baserun.submission_service.SubmitSpan(span_request)
            except Exception as e:
                if hasattr(e, "details"):
                    logger.warning(f"Failed to submit span to Baserun: {e.details()}")
                else:
                    logger.warning(f"Failed to submit span to Baserun: {e}")
