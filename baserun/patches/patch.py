import functools
import inspect
from typing import Any, Callable, List
import time


class Patch:
    def __init__(self, module: Any, symbols: List[str], resolver: Callable, log: Callable) -> None:
        self.resolver = resolver
        self.log = log
        for symbol in symbols:
            symbol_parts = symbol.split('.')
            original = functools.reduce(getattr, symbol_parts, module)

            if len(symbol_parts) == 1:
                setattr(module, symbol_parts[0], self._patched_method(symbol, original))
            else:
                setattr(functools.reduce(getattr, symbol_parts[:-1], module), symbol_parts[-1], self._patched_method(symbol, original))

    def _patched_method(self, symbol: str, original_method):
        unwrapped_method = original_method
        while hasattr(unwrapped_method, '__wrapped__'):
            unwrapped_method = unwrapped_method.__wrapped__

        if inspect.iscoroutinefunction(unwrapped_method):
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                response = None
                error = None
                try:
                    response = await original_method(*args, **kwargs)
                    return response
                except Exception as e:
                    error = e
                    raise
                finally:
                    end_time = time.time()
                    log_entry = self.resolver(symbol, args, kwargs, start_time, end_time, response, error)
                    self.log(log_entry)
            return async_wrapper
        else:
            def wrapper(*args, **kwargs):
                start_time = time.time()
                response = None
                error = None
                try:
                    response = original_method(*args, **kwargs)
                    return response
                except Exception as e:
                    error = e
                    raise
                finally:
                    end_time = time.time()
                    log_entry = self.resolver(symbol, args, kwargs, start_time, end_time, response, error)
                    self.log(log_entry)
            return wrapper
