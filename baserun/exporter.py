from typing import Sequence, Any

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter

from baserun.instrumentation.openai import WRAPPED_METHODS
from baserun.instrumentation.span_attributes import SpanAttributes
from baserun.v1.baserun_pb2 import Status, Message, Span, SubmitSpanRequest


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
            if span.name not in [method["span_name"] for method in WRAPPED_METHODS]:
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

            # Trace IDs are huge integers so they must be encoded as bytes
            trace_id_int = span.context.trace_id
            trace_id = trace_id_int.to_bytes(
                (trace_id_int.bit_length() + 7) // 8, "big"
            )

            span_message = Span(
                trace_id=trace_id,
                span_id=span.context.span_id,
                name=span.name,
                start_time=span.start_time,
                end_time=span.end_time,
                status=status,
                vendor=span.attributes.get(SpanAttributes.LLM_VENDOR, ""),
                request_type=span.attributes.get(SpanAttributes.LLM_REQUEST_TYPE, ""),
                api_base=span.attributes.get(SpanAttributes.OPENAI_API_BASE, ""),
                api_type=span.attributes.get(SpanAttributes.OPENAI_API_TYPE, ""),
                model=span.attributes.get(SpanAttributes.LLM_REQUEST_MODEL, ""),
                temperature=span.attributes.get(SpanAttributes.LLM_TEMPERATURE, 1),
                top_p=span.attributes.get(SpanAttributes.LLM_TOP_P, 1),
                frequency_penalty=span.attributes.get(
                    SpanAttributes.LLM_FREQUENCY_PENALTY, 0
                ),
                presence_penalty=span.attributes.get(
                    SpanAttributes.LLM_PRESENCE_PENALTY, 0
                ),
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
                completions=completions,
            )
            span_request = SubmitSpanRequest(span=span_message)
            Baserun._submit_span_stub.SubmitSpan(span_request)

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        pass
