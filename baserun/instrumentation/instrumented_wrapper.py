import logging
from typing import TYPE_CHECKING

from opentelemetry.sdk.trace import _Span
from opentelemetry.trace import (
    SpanKind,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def setup_span(span_name: str, parent_span: _Span) -> dict:
    if not parent_span or not parent_span.is_recording():
        return {"name": f"baserun.{span_name}", "kind": SpanKind.CLIENT, "attributes": {}}

    return {
        "name": f"baserun.{span_name}",
        "kind": SpanKind.CLIENT,
        "attributes": parent_span.attributes,
    }
