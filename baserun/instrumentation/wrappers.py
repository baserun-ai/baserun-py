import logging
from types import GeneratorType

from opentelemetry import context as context_api, trace
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY, get_value
from opentelemetry.sdk.trace import Span
from opentelemetry.trace import SpanKind, Status, StatusCode

from baserun.instrumentation.base_instrumentor import BaseInstrumentor
from baserun.instrumentation.span_attributes import SpanAttributes

logger = logging.getLogger(__name__)


def with_tracer_wrapper(func):
    def with_tracer(tracer, to_wrap, instrumentor):
        def wrapper(wrapped, instance, args, kwargs):
            # prevent double wrapping
            if hasattr(wrapped, "__wrapped__"):
                return wrapped(*args, **kwargs)

            return func(tracer, to_wrap, instrumentor, wrapped, instance, args, kwargs)

        return wrapper

    return with_tracer


def generator_wrapper(original_generator: GeneratorType, span: Span):
    for value in original_generator:
        # Currently we only support one choice in streaming responses
        prefix = f"{SpanAttributes.LLM_COMPLETIONS}.0"

        role = value.choices[0].delta.get("role")
        if role:
            span.set_attribute(f"{prefix}.role", role)

        new_content = value.choices[0].delta.get("content")
        if new_content:
            span.set_attribute(
                f"{prefix}.content",
                span.attributes.get(f"{prefix}.content", "") + new_content,
            )

        if new_content is None or value.choices[0].finish_reason:
            span.end()

        yield value


@with_tracer_wrapper
def instrumented_wrapper(
    tracer, to_wrap, instrumentor: BaseInstrumentor, wrapped, instance, args, kwargs
):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    name = to_wrap.get("span_name")

    # Start new span
    span = tracer.start_span(
        name,
        kind=SpanKind.CLIENT,
        attributes={
            SpanAttributes.LLM_REQUEST_TYPE: name.split(".")[-1],
        },
    )

    # Activate the span in the current context, but don't end it automatically
    with trace.use_span(span, end_on_exit=False):
        span.set_attribute(
            SpanAttributes.BASERUN_RUN_ID, get_value(SpanAttributes.BASERUN_RUN_ID)
        )

        # Capture request attributes
        # noinspection PyBroadException
        try:
            instrumentor.set_request_attributes(span, kwargs)

            span.set_attribute(SpanAttributes.LLM_REQUEST_MODEL, kwargs.get("model"))

            max_tokens = kwargs.get("max_tokens", kwargs.get("max_tokens_to_sample"))
            if max_tokens is not None:
                span.set_attribute(SpanAttributes.LLM_REQUEST_MAX_TOKENS, max_tokens)
        except Exception as e:
            logger.warning(
                f"Failed to set input attributes for Baserun span, error: {e}"
            )

        # Actually call the wrapped method
        response = wrapped(*args, **kwargs)

        # If this is a streaming response, wrap it, so we can capture each chunk
        if isinstance(response, GeneratorType):
            wrapped_response = generator_wrapper(response, span)
            return wrapped_response

        # If it's a full response capture the response attributes
        if response:
            # noinspection PyBroadException
            try:
                span.set_status(Status(StatusCode.OK))
                instrumentor.set_response_attributes(span, response)
            except Exception as e:
                import pdb

                pdb.set_trace()
                logger.warning(
                    f"Failed to set response attributes for Baserun span, error: {e}"
                )
        else:
            # Not sure when this could happen?
            span.set_status(
                Status(description="No response received", status_code=StatusCode.UNSET)
            )

    # End the span and return the response
    span.end()
    return response
