from typing import Callable

from opentelemetry import trace
from opentelemetry.sdk.trace import _Span
from opentelemetry.trace import get_current_span, SpanKind


def baserun_thread_wrapper(target: Callable, *args, **kwargs):
    """Given a target function intended to be run in a new thread, wrap the target in a function and start a new
    parent span which propagates attributes from the parent thread's parent span."""
    parent_span: _Span = get_current_span()
    if "results" in kwargs:
        results = kwargs.pop("results")
    else:
        results = None

    def wrapper():
        tracer_provider = trace.get_tracer_provider()
        tracer = tracer_provider.get_tracer("baserun")
        span = tracer.start_span(
            f"baserun.parent.inner_thread",
            kind=SpanKind.CLIENT,
            attributes=parent_span.attributes,
        )
        try:
            output = target(*args, **kwargs)
            if results is not None:
                results.append(output)
            return output
        finally:
            if span.is_recording():
                span.end()

    return wrapper
