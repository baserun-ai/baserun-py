import functools
import inspect
from typing import Any, Callable, List
import time


class Patch:
    def __init__(self, module: Any, symbols: List[str], resolver: Callable, log: Callable, is_streaming: Callable, collect_streamed_response: Callable) -> None:
        self.resolver = resolver
        self.log = log
        self.is_streaming = is_streaming
        self.collect_streamed_response = collect_streamed_response

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
                is_stream = self.is_streaming(symbol, args, kwargs)
                start_time = time.time()
                response = None
                error = None

                if is_stream:
                    async def streaming_async_wrapper():
                        stream_response = None
                        stream_error = None
                        try:
                            collected_response = None
                            async for chunk in await original_method(*args, **kwargs):
                                collected_response = self.collect_streamed_response(symbol, collected_response, chunk)
                                yield chunk
                            stream_response = collected_response
                        except Exception as stream_e:
                            stream_error = stream_e
                            raise
                        finally:
                            stream_end_time = time.time()
                            stream_log_entry = self.resolver(symbol, args, kwargs, start_time, stream_end_time, stream_response, stream_error)
                            self.log(stream_log_entry)

                    return streaming_async_wrapper()
                else:
                    try:
                        response = await original_method(*args, **kwargs)
                    except Exception as e:
                        error = e
                        raise
                    finally:
                        end_time = time.time()
                        log_entry = self.resolver(symbol, args, kwargs, start_time, end_time, response, error)
                        self.log(log_entry)

                    return response

            return async_wrapper
        else:
            def wrapper(*args, **kwargs):
                is_stream = self.is_streaming(symbol, args, kwargs)
                start_time = time.time()
                response = None
                error = None

                if is_stream:
                    def streaming_wrapper():
                        stream_response = None
                        stream_error = None
                        try:
                            collected_response = None
                            for chunk in original_method(*args, **kwargs):
                                collected_response = self.collect_streamed_response(symbol, collected_response, chunk)
                                yield chunk
                            stream_response = collected_response
                        except Exception as stream_e:
                            stream_error = stream_e
                            raise
                        finally:
                            stream_end_time = time.time()
                            stream_log_entry = self.resolver(symbol, args, kwargs, start_time, stream_end_time, stream_response, stream_error)
                            self.log(stream_log_entry)

                    return streaming_wrapper()
                else:
                    try:
                        response = original_method(*args, **kwargs)
                    except Exception as e:
                        error = e
                        raise
                    finally:
                        end_time = time.time()
                        log_entry = self.resolver(symbol, args, kwargs, start_time, end_time, response, error)
                        self.log(log_entry)

                    return response
            return wrapper
