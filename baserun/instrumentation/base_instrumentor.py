import collections.abc
import inspect
from abc import abstractmethod
from typing import Any, Callable

from opentelemetry.instrumentation.instrumentor import (
    BaseInstrumentor as OTelInstrumentor,
)
from opentelemetry.sdk.trace import Span, _Span

from baserun.instrumentation.instrumented_wrapper import (
    instrumented_wrapper,
    async_instrumented_wrapper,
)


class BaseInstrumentor(OTelInstrumentor):
    original_methods: dict[str, Callable] = None

    @staticmethod
    @abstractmethod
    def wrapped_methods() -> list[dict[str, Any]]:
        pass

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
    def generator_wrapper(original_generator: collections.abc.Iterator, span: _Span):
        pass

    @staticmethod
    @abstractmethod
    def async_generator_wrapper(
        original_generator: collections.abc.AsyncIterator, span: _Span
    ):
        pass

    def _instrument(self, **kwargs):
        if not self.wrapped_methods():
            return

        if not BaseInstrumentor.original_methods:
            BaseInstrumentor.original_methods = {}

        for method_spec in self.wrapped_methods():
            original_method = method_spec["function"]
            original_class = method_spec["class"]

            BaseInstrumentor.original_methods[
                f"{original_class.__module__}.{original_class.__name__}.{original_method.__name__}"
            ] = original_method

            unwrapped_method = original_method
            while hasattr(unwrapped_method, "__wrapped__"):
                unwrapped_method = unwrapped_method.__wrapped__

            if inspect.iscoroutinefunction(unwrapped_method):
                wrapper = async_instrumented_wrapper(
                    original_method, self, method_spec["span_name"]
                )
            else:
                wrapper = instrumented_wrapper(
                    original_method, self, method_spec["span_name"]
                )
            setattr(original_class, original_method.__name__, wrapper)

    def _uninstrument(self, **kwargs):
        if not self.wrapped_methods():
            return

        for method_spec in self.wrapped_methods():
            original_method = method_spec["function"]
            original_class = method_spec["class"]
            setattr(original_class, original_method.__name__, original_method)
