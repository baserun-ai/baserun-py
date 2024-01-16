import time
from enum import Enum, auto
from typing import Union

from opentelemetry.sdk.trace import _Span
from opentelemetry.trace import get_current_span

from baserun.instrumentation.span_attributes import BASERUN_SESSION_ID


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
    span: _Span = get_current_span()
    if not span.is_recording():
        return
    session_id = span.attributes.get(BASERUN_SESSION_ID)
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
