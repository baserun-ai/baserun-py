import time
from enum import Enum, auto
from typing import Union

from opentelemetry.sdk.trace import _Span
from opentelemetry.trace import get_current_span

from .constants import SESSION_SPAN_NAME
from .instrumentation.span_attributes import SpanAttributes


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


def get_session_id() -> Union[str, None]:
    parent_span: _Span = get_current_span()
    session_id = ""
    if parent_span.is_recording() and parent_span.name.startswith(SESSION_SPAN_NAME):
        session_id = parent_span.attributes.get(SpanAttributes.BASERUN_SESSION_ID)

    return session_id


def memoize_for_time(seconds):
    def decorator(func):
        cache = {}

        def wrapper(*args, **kwargs):
            now = time.time()
            if "timestamp" in cache and now - cache["timestamp"] < seconds:
                return cache["value"]

            result = func(*args, **kwargs)
            cache["value"] = result
            cache["timestamp"] = now
            return result

        return wrapper

    return decorator
