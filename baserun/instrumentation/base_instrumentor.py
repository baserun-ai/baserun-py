from abc import abstractmethod
from types import GeneratorType
from typing import Any

from opentelemetry.instrumentation.instrumentor import (
    BaseInstrumentor as OTelInstrumentor,
)
from opentelemetry.sdk.trace import Span

from baserun.instrumentation.instrumented_wrapper import instrumented_wrapper


class BaseInstrumentor(OTelInstrumentor):
    WRAPPED_METHODS: list[dict[str, Any]] = None

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

    def _instrument(self, **kwargs):
        if not self.WRAPPED_METHODS:
            return

        for method_spec in self.WRAPPED_METHODS:
            original_method = method_spec["function"]
            original_class = method_spec["class"]
            wrapper = instrumented_wrapper(
                original_method, self, method_spec["span_name"]
            )
            setattr(original_class, original_method.__name__, wrapper)

    def _uninstrument(self, **kwargs):
        if not self.WRAPPED_METHODS:
            return

        for method_spec in self.WRAPPED_METHODS:
            original_method = method_spec["function"]
            original_class = method_spec["class"]
            setattr(original_class, original_method.__name__, original_method)
