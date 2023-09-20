from abc import abstractmethod
from typing import Any

from opentelemetry.instrumentation.instrumentor import (
    BaseInstrumentor as OTelInstrumentor,
)
from opentelemetry.sdk.trace import Span


class BaseInstrumentor(OTelInstrumentor):
    @abstractmethod
    def set_request_attributes(self, span: Span, kwargs: dict[str, Any]):
        pass

    @abstractmethod
    def set_response_attributes(self, span: Span, response: Any):
        pass
