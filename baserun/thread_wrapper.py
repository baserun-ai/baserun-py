from typing import Callable

from opentelemetry import trace
from opentelemetry.sdk.trace import _Span
from opentelemetry.trace import get_current_span, SpanKind

from baserun import Baserun


def baserun_thread_wrapper(target: Callable):
    """Given a target function intended to be run in a new thread, wrap the target in a function and start a new
    parent span which propagates attributes from the parent thread's parent span."""
    if not Baserun._initialized:
        return target

    parent_span: _Span = get_current_span()

    def wrapper(*args, **kwargs):
        tracer_provider = trace.get_tracer_provider()
        tracer = tracer_provider.get_tracer("baserun")
        span = tracer.start_span(
            f"baserun.parent.inner_thread",
            kind=SpanKind.CLIENT,
            attributes=parent_span.attributes,
        )
        try:
            return target(*args, **kwargs)
        finally:
            if span.is_recording():
                span.end()

    return wrapper
