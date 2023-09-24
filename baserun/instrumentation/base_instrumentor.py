from abc import abstractmethod
from types import GeneratorType
from typing import Any

from opentelemetry.instrumentation.instrumentor import (
    BaseInstrumentor as OTelInstrumentor,
)
from opentelemetry.sdk.trace import Span


class BaseInstrumentor(OTelInstrumentor):
    @staticmethod
    @abstractmethod
    def set_request_attributes(span: Span, kwargs: dict[str, Any]):
        pass

    @staticmethod
    @abstractmethod
    def set_response_attributes(span: Span, response: Any):
        pass

    @staticmethod
    @abstractmethod
    def generator_wrapper(original_generator: GeneratorType, span: Span):
        pass
