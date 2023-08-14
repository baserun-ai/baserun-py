import inspect
import json
import uuid
import os
import requests
import time
import threading
from typing import Callable, Dict, Optional, Union
from urllib.parse import urlparse
import warnings
from .helpers import BaserunStepType, TraceType
from .openai import monkey_patch_openai


class Baserun:
    _initialized = False
    _trace_id = None
    _traces = []
    _buffer = []
    _buffer_lock = threading.Lock()
    _api_base_url = None
    _api_key = None

    @staticmethod
    def init(api_base_url: str = "https://baserun.ai/api/v1") -> None:
        api_key = os.environ.get('BASERUN_API_KEY')
        if not api_key:
            raise ValueError("Baserun API key is missing. Ensure the BASERUN_API_KEY environment variable is set.")

        if Baserun._initialized:
            warnings.warn("Baserun has already been initialized. Additional calls to init will be ignored.")
            return

        Baserun._api_base_url = api_base_url
        Baserun._api_key = api_key
        Baserun._initialized = True

        monkey_patch_openai(Baserun._append_to_buffer)

    @staticmethod
    def _trace(func: Callable, trace_type: TraceType, metadata: Optional[Dict] = None):
        if inspect.iscoroutinefunction(func):
            async def wrapper(*args, **kwargs):
                if not Baserun._initialized or Baserun._trace_id:
                    return await func(*args, **kwargs)

                test_name = func.__name__
                test_inputs = []
                for input_name, input_value in kwargs.items():
                    if inspect.iscoroutine(input_value):
                        input_result = input_value.__name__
                    else:
                        input_result = input_value
                    test_inputs.append(f"{input_name}: {input_result}")

                test_execution_id = str(uuid.uuid4())
                Baserun._trace_id = test_execution_id
                Baserun._buffer = []
                start_time = time.time()

                try:
                    result = await func(*args, **kwargs)
                    end_time = time.time()
                    Baserun.store_trace({
                        'type': trace_type,
                        'testName': test_name,
                        'testInputs': test_inputs,
                        'id': test_execution_id,
                        'result': str(result) if result is not None else '',
                        'startTimestamp': start_time,
                        'completionTimestamp': end_time,
                        "steps": Baserun._buffer,
                        "metadata": metadata,
                    })
                    return result
                except Exception as e:
                    end_time = time.time()
                    Baserun.store_trace({
                        'type': trace_type,
                        'testName': test_name,
                        'testInputs': test_inputs,
                        'id': test_execution_id,
                        'error': str(e),
                        'startTimestamp': start_time,
                        'completionTimestamp': end_time,
                        "steps": Baserun._buffer,
                        "metadata": metadata,
                    })
                    raise e
                finally:
                    if trace_type == TraceType.PRODUCTION:
                        Baserun.flush()

                    Baserun._buffer = []
                    Baserun._trace_id = None
        else:
            def wrapper(*args, **kwargs):
                if not Baserun._initialized or Baserun._trace_id:
                    return func(*args, **kwargs)

                test_name = func.__name__
                test_inputs = []
                for input_name, input_value in kwargs.items():
                    test_inputs.append(f"{input_name}: {input_value}")

                test_execution_id = str(uuid.uuid4())
                Baserun._trace_id = test_execution_id
                Baserun._buffer = []
                start_time = time.time()

                try:
                    result = func(*args, **kwargs)
                    end_time = time.time()
                    Baserun.store_trace({
                        'type': trace_type,
                        'testName': test_name,
                        'testInputs': test_inputs,
                        'id': test_execution_id,
                        'result': str(result) if result is not None else '',
                        'startTimestamp': start_time,
                        'completionTimestamp': end_time,
                        "steps": Baserun._buffer,
                        "metadata": metadata,
                    })
                    return result
                except Exception as e:
                    end_time = time.time()
                    Baserun.store_trace({
                        'type': trace_type,
                        'testName': test_name,
                        'testInputs': test_inputs,
                        'id': test_execution_id,
                        'error': str(e),
                        'startTimestamp': start_time,
                        'completionTimestamp': end_time,
                        "steps": Baserun._buffer,
                        "metadata": metadata,
                    })
                    raise e
                finally:
                    if trace_type == TraceType.PRODUCTION:
                        Baserun.flush()

                    Baserun._buffer = []
                    Baserun._trace_id = None

        return wrapper

    @staticmethod
    def trace(func: Callable, metadata: Optional[Dict] = None):
        return Baserun._trace(func, TraceType.PRODUCTION, metadata)

    @staticmethod
    def test(func: Callable, metadata: Optional[Dict] = None):
        return Baserun._trace(func, TraceType.TEST, metadata)

    @staticmethod
    def log(name: str, payload: Union[str, Dict]):
        if not Baserun._initialized:
            return

        if not Baserun._trace_id:
            warnings.warn("baserun.log was called outside of a Baserun decorated trace. The log will be ignored.")
            return

        log_entry = {
            "stepType": BaserunStepType.LOG.name.lower(),
            "name": name,
            "payload": payload,
            "timestamp": time.time(),
        }

        Baserun._append_to_buffer(log_entry)

    @staticmethod
    def store_trace(trace_data: Dict):
        Baserun._traces.append(trace_data)

    @staticmethod
    def flush():
        if not Baserun._initialized:
            warnings.warn("Baserun has not been initialized. No data will be flushed.")
            return

        if not Baserun._traces:
            return

        headers = {"Authorization": f"Bearer {Baserun._api_key}"}

        try:
            if all(trace.get('type') == TraceType.TEST for trace in Baserun._traces):
                response = requests.post(f"{Baserun._api_base_url}/runs", json=json.loads(json.dumps({"testExecutions": Baserun._traces}, default=str)), headers=headers)
                response.raise_for_status()

                response_data = response.json()
                test_run_id = response_data['id']
                parsed_url = urlparse(Baserun._api_base_url)
                url = f"{parsed_url.scheme}://{parsed_url.netloc}/runs/{test_run_id}"
                return url
            elif all(trace.get('type') == TraceType.PRODUCTION for trace in Baserun._traces):
                response = requests.post(f"{Baserun._api_base_url}/traces", json=json.loads(json.dumps({"traces": Baserun._traces}, default=str)), headers=headers)
                response.raise_for_status()
                return
            else:
                warnings.warn("Inconsistent trace types, skipping Baserun upload")
        except requests.RequestException as e:
            warnings.warn(f"Failed to upload results to Baserun: {str(e)}")

        finally:
            Baserun._traces = []

    @staticmethod
    def _append_to_buffer(log_entry: Dict):
        with Baserun._buffer_lock:
            Baserun._buffer.append(log_entry)
