import logging
from types import GeneratorType
from typing import Callable, TYPE_CHECKING

from opentelemetry import context as context_api, trace
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.sdk.trace import _Span
from opentelemetry.trace import (
    SpanKind,
    Status,
    StatusCode,
    get_current_span,
)

from baserun.instrumentation.span_attributes import SpanAttributes

if TYPE_CHECKING:
    from baserun.instrumentation.base_instrumentor import BaseInstrumentor

logger = logging.getLogger(__name__)


def instrumented_wrapper(
    wrapped_fn: Callable, instrumentor: "BaseInstrumentor", span_name: str
):
    """Generates a function (`instrumented_function`) which instruments the original function (`wrapped_fn`)"""

    def instrumented_function(*args, **kwargs):
        """Replacement for the original function (`wrapped_fn`). Will perform instrumentation, call `wrapped_fn`,
        perform more instrumentation, and then return the result of the previous call to `wrapped_fn`.
        """
        from baserun import Baserun

        tracer_provider = trace.get_tracer_provider()
        tracer = tracer_provider.get_tracer("baserun")

        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped_fn(*args, **kwargs)

        parent_span: _Span = get_current_span()
        request_type = span_name.split(".")[-1]
        parent_span.set_attribute(SpanAttributes.LLM_REQUEST_TYPE, request_type)

        # Start new span
        span = tracer.start_span(
            f"baserun.{span_name}",
            kind=SpanKind.CLIENT,
            attributes=parent_span.attributes,
        )

        error_to_reraise = None

        auto_end_span = True
        try:
            # Activate the span in the current context, but don't end it automatically
            with trace.use_span(span, end_on_exit=False):
                # Capture request attributes
                # noinspection PyBroadException
                try:
                    instrumentor.set_request_attributes(span, kwargs)

                    span.set_attribute(
                        SpanAttributes.LLM_REQUEST_MODEL, kwargs.get("model")
                    )

                    max_tokens = kwargs.get(
                        "max_tokens", kwargs.get("max_tokens_to_sample")
                    )
                    if max_tokens is not None:
                        span.set_attribute(
                            SpanAttributes.LLM_REQUEST_MAX_TOKENS, max_tokens
                        )
                except Exception as e:
                    logger.warning(
                        f"Failed to set input attributes for Baserun span, error: {e}"
                    )

                # Actually call the wrapped method
                try:
                    response = wrapped_fn(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                except Exception as e:
                    span.set_status(
                        Status(
                            description=str(e),
                            status_code=StatusCode.ERROR,
                        )
                    )
                    raise e

                # If this is a streaming response, wrap it, so we can capture each chunk
                if isinstance(response, GeneratorType):
                    wrapped_response = instrumentor.generator_wrapper(response, span)
                    # The span will be ended inside the generator once it's finished
                    auto_end_span = False
                    return wrapped_response

                run = Baserun.current_run()
                if not run:
                    logger.warning(
                        "Baserun data not propagated correctly, cannot send data"
                    )
                    return response

                span.set_attribute(
                    SpanAttributes.BASERUN_RUN, Baserun.serialize_run(run)
                )

                # If it's a full response capture the response attributes
                if response:
                    # noinspection PyBroadException
                    try:
                        instrumentor.set_response_attributes(span, response)
                    except Exception as e:
                        logger.warning(
                            f"Failed to set response attributes for Baserun span, error: {e}"
                        )
                else:
                    # Not sure when this could happen?
                    span.set_status(
                        Status(
                            description="No response received",
                            status_code=StatusCode.UNSET,
                        )
                    )
        finally:
            if auto_end_span:
                span.end()

        return response

    return instrumented_function
