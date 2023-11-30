import logging
import traceback
from collections.abc import AsyncIterator, Iterator
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

from baserun.constants import UNTRACED_SPAN_PARENT_NAME
from baserun.helpers import get_session_id
from baserun.instrumentation.span_attributes import SpanAttributes
from baserun.v1.baserun_pb2 import Run

if TYPE_CHECKING:
    from baserun.instrumentation.base_instrumentor import BaseInstrumentor

logger = logging.getLogger(__name__)


def async_instrumented_wrapper(
    wrapped_fn: Callable, instrumentor: "BaseInstrumentor", span_name: str
):
    """Generates a function (`instrumented_function`) which instruments the original function (`wrapped_fn`)"""

    async def instrumented_function(*args, **kwargs):
        from baserun import Baserun

        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return await wrapped_fn(*args, **kwargs)

        tracer_provider = trace.get_tracer_provider()
        tracer = tracer_provider.get_tracer("baserun")

        parent_span: _Span = get_current_span()
        if not parent_span.is_recording() and not Baserun.current_test_suite:
            run = Baserun.get_or_create_current_run(
                name=UNTRACED_SPAN_PARENT_NAME,
                trace_type=Run.RunType.RUN_TYPE_PRODUCTION,
            )
            parent_span = tracer.start_as_current_span(
                UNTRACED_SPAN_PARENT_NAME,
                kind=SpanKind.CLIENT,
                attributes={
                    SpanAttributes.BASERUN_RUN: Baserun.serialize_run(run),
                },
            )

        session_id = get_session_id()
        span = tracer.start_span(
            **setup_span(span_name=span_name, parent_span=parent_span)
        )
        if session_id:
            span.set_attribute(SpanAttributes.BASERUN_SESSION_ID, session_id)

        auto_end_span = True
        try:
            # Activate the span in the current context, but don't end it automatically
            with trace.use_span(span, end_on_exit=False):
                # Capture request attributes
                set_request_attributes(
                    instrumentor=instrumentor, span=span, kwargs=kwargs
                )

                # Actually call the wrapped method
                try:
                    response = await wrapped_fn(*args, **kwargs)

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
                if isinstance(response, AsyncIterator):
                    wrapped_response = instrumentor.async_generator_wrapper(
                        response, span
                    )
                    # The span will be ended inside the generator once it's finished
                    auto_end_span = False
                    return wrapped_response

                handle_response(instrumentor, span, response)
        finally:
            if auto_end_span:
                if span.is_recording():
                    span.end()

                if (
                    parent_span.name == UNTRACED_SPAN_PARENT_NAME
                    and parent_span.is_recording()
                    and not Baserun.current_test_suite
                ):
                    parent_span.end()

        return response

    return instrumented_function


def instrumented_wrapper(
    wrapped_fn: Callable, instrumentor: "BaseInstrumentor", span_name: str = None
):
    """Generates a function (`instrumented_function`) which instruments the original function (`wrapped_fn`)"""

    def instrumented_function(*args, **kwargs):
        """Replacement for the original function (`wrapped_fn`). Will perform instrumentation, call `wrapped_fn`,
        perform more instrumentation, and then return the result of the previous call to `wrapped_fn`.
        """
        from baserun import Baserun

        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped_fn(*args, **kwargs)

        tracer_provider = trace.get_tracer_provider()
        tracer = tracer_provider.get_tracer("baserun")

        parent_span: _Span = get_current_span()
        # If a call is made outside of a traced function we need to create a parent
        if not parent_span.is_recording():
            run = Baserun.get_or_create_current_run(
                name=wrapped_fn.__name__,
                trace_type=Run.RunType.RUN_TYPE_PRODUCTION,
            )
            parent_span = tracer.start_as_current_span(
                UNTRACED_SPAN_PARENT_NAME,
                kind=SpanKind.CLIENT,
                attributes={
                    SpanAttributes.BASERUN_RUN: Baserun.serialize_run(run),
                },
            )

        session_id = get_session_id()
        span = tracer.start_span(
            **setup_span(span_name=span_name, parent_span=parent_span)
        )
        if session_id:
            span.set_attribute(SpanAttributes.BASERUN_SESSION_ID, session_id)

        auto_end_span = True
        try:
            # Activate the span in the current context, but don't end it automatically
            with trace.use_span(span, end_on_exit=False):
                # Capture request attributes
                set_request_attributes(
                    instrumentor=instrumentor, span=span, kwargs=kwargs
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
                if isinstance(response, Iterator):
                    wrapped_response = instrumentor.generator_wrapper(response, span)
                    # The span will be ended inside the generator once it's finished
                    auto_end_span = False
                    return wrapped_response

                handle_response(instrumentor, span, response)
        finally:
            if auto_end_span:
                if span.is_recording():
                    span.end()

                if (
                    parent_span.name == UNTRACED_SPAN_PARENT_NAME
                    and parent_span.is_recording()
                    and not Baserun.current_test_suite
                ):
                    parent_span.end()

        return response

    return instrumented_function


def setup_span(span_name: str, parent_span: _Span) -> dict:
    request_type = span_name.split(".")[-1]

    parent_span.set_attribute(SpanAttributes.LLM_REQUEST_TYPE, request_type)

    return {
        "name": f"baserun.{span_name}",
        "kind": SpanKind.CLIENT,
        "attributes": parent_span.attributes,
    }


def set_request_attributes(
    instrumentor: "BaseInstrumentor", span: _Span, kwargs: dict
) -> _Span:
    try:
        instrumentor.set_request_attributes(span, kwargs)
    except Exception as e:
        traceback.print_exception(e)
        logger.warning(f"Failed to set input attributes for Baserun span, error: {e}")

    return span


def handle_response(instrumentor: "BaseInstrumentor", span: _Span, response):
    from baserun import Baserun

    run = Baserun.current_run()
    if not run:
        logger.warning("Baserun data not propagated correctly, cannot send data")
        return response

    span.set_attribute(SpanAttributes.BASERUN_RUN, Baserun.serialize_run(run))

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
        # This will happen if the user doesn't return anything from their traced function
        span.set_status(
            Status(
                description="No response received",
                status_code=StatusCode.UNSET,
            )
        )

    return span
