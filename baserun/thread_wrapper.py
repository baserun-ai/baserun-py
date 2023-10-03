from typing import Callable

from opentelemetry import trace
from opentelemetry.sdk.trace import Span
from opentelemetry.trace import get_current_span, SpanKind

from baserun import Baserun


def baserun_thread_wrapper(target: Callable):
    """Given a target function intended to be run in a new thread, wrap the target in a function and start a new
    parent span which propagates attributes from the parent thread's parent span."""
    if not Baserun._initialized:
        return target

    parent_span: Span = get_current_span()

    def wrapper(*args, **kwargs):
        tracer_provider = trace.get_tracer_provider()
        tracer = tracer_provider.get_tracer("baserun")
        with tracer.start_as_current_span(
                f"baserun.parent.inner_thread",
                kind=SpanKind.CLIENT,
                attributes=parent_span.attributes,
        ) as span:
            return target(*args, **kwargs)

    return wrapper
