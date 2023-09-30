from enum import Enum, auto
from typing import Callable

from opentelemetry import trace
from opentelemetry.sdk.trace import _Span
from opentelemetry.trace import get_current_span, SpanKind


class BaserunProvider(Enum):
    ANTHROPIC = auto()
    OPENAI = auto()


class BaserunType(Enum):
    CHAT = auto()
    COMPLETION = auto()


class BaserunStepType(Enum):
    LOG = auto()
    AUTO_LLM = auto()
    CUSTOM_LLM = auto()


def baserun_threaded_wrapper(
    target: Callable, results: list, thread_args=None, thread_kwargs=None
):
    """Given a target function intended to be run in a new thread, wrap the target in a function and start a new
    parent span which propagates attributes from the parent thread's parent span."""
    thread_args = thread_args or []
    thread_kwargs = thread_kwargs or {}
    parent_span: _Span = get_current_span()

    def wrapper():
        tracer_provider = trace.get_tracer_provider()
        tracer = tracer_provider.get_tracer("baserun")
        with tracer.start_as_current_span(
            f"baserun.parent.inner_thread",
            kind=SpanKind.CLIENT,
            attributes=parent_span.attributes,
        ):
            output = target(*thread_args, **thread_kwargs)
            results.append(output)
            return output

    return wrapper
