import json
import time
from enum import Enum, auto
from typing import Union

from opentelemetry.sdk.trace import _Span
from opentelemetry.trace import Span, get_current_span

from baserun.instrumentation.span_attributes import BASERUN_SESSION_ID
from baserun.v1.baserun_pb2 import Run


class BaserunProvider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"
    # not really a provider but we can treat it as such and it's handy
    LLAMA_INDEX = "llama_index"


class BaserunType(str, Enum):
    CHAT = "chat"
    COMPLETION = "completion"


class BaserunStepType(Enum):
    LOG = auto()
    AUTO_LLM = auto()
    CUSTOM_LLM = auto()


def get_session_id() -> Union[str, None]:
    span: Union[_Span, Span] = get_current_span()
    # Of type NonRecordingSpan
    if not span.is_recording():
        return None

    if not hasattr(span, "attributes"):
        # Of type Span, we don't handle this, as it's probably not ours
        return None

    # Of type _Span (SDK's version of Span)
    session_id: str = span.attributes.get(BASERUN_SESSION_ID)
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

        def clear_cache():
            cache.clear()

        wrapper.clear_cache = clear_cache

        return wrapper

    return decorator


def patch_run_for_metadata():
    """
    Allow assignment of any JSON-ifiable object to the `metadata` parameter
    """
    original_setattr = Run.__setattr__

    def jsonifiable_setattr(self, name, value):
        if name == "metadata" and not isinstance(value, str):
            try:
                value = json.dumps(value)
            except TypeError:
                value = str(value)

        # Call the original implementation to actually set the attribute
        original_setattr(self, name, value)

    Run.__setattr__ = jsonifiable_setattr
