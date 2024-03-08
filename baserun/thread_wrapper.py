from typing import Callable, Union

from opentelemetry import trace
from opentelemetry.sdk.trace import _Span
from opentelemetry.trace import Span, SpanKind, get_current_span

from baserun import Baserun

from .constants import INNER_THREAD_SPAN_PARENT_NAME


def baserun_thread_wrapper(target: Callable):
    """Given a target function intended to be run in a new thread, wrap the target in a function and start a new
    parent span which propagates attributes from the parent thread's parent span."""
    if not Baserun.initialized:
        return target

    parent_span: Union[Span, _Span] = get_current_span()

    def wrapper(*args, **kwargs):
        tracer_provider = trace.get_tracer_provider()
        tracer = tracer_provider.get_tracer("baserun")
        attributes = {}
        if isinstance(parent_span, _Span):
            attributes = parent_span.attributes

        span = tracer.start_span(
            INNER_THREAD_SPAN_PARENT_NAME,
            kind=SpanKind.CLIENT,
            attributes=attributes,
        )
        with trace.use_span(span, end_on_exit=False):
            try:
                return target(*args, **kwargs)
            finally:
                if span.is_recording():
                    span.end()

    return wrapper
