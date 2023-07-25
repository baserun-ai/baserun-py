import uuid
import os
import requests
import threading
import warnings
import time

_thread_local = threading.local()


class Baserun:
    _initialized = False
    _tests = []
    _api_url = None
    _api_key = None

    @staticmethod
    def init(api_url: str = "https://baserun.ai/api/runs") -> None:
        api_key = os.environ.get('BASERUN_API_KEY')
        if not api_key:
            raise ValueError("Baserun API key is missing. Ensure the BASERUN_API_KEY environment variable is set.")

        if Baserun._initialized:
            warnings.warn("Baserun has already been initialized. Additional calls to init will be ignored.")
            return

        Baserun._api_url = api_url
        Baserun._api_key = api_key
        Baserun._initialized = True

    @staticmethod
    def test(func):
        def wrapper(*args, **kwargs):
            if not Baserun._initialized:
                return func(*args, **kwargs)

            test_name = func.__name__
            test_id = str(uuid.uuid4())
            _thread_local.baserun_id = test_id
            _thread_local.buffer = []
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
            except Exception as e:
                end_time = time.time()
                Baserun.store_test({
                    'name': test_name,
                    'id': test_id,
                    'error': str(e),
                    'startTimestamp': start_time,
                    'completionTimestamp': end_time,
                    "steps": _thread_local.buffer
                })
                _thread_local.buffer = []
                del _thread_local.baserun_id
                raise e

            end_time = time.time()
            Baserun.store_test({
                'name': test_name,
                'id': test_id,
                'result': str(result),
                'startTimestamp': start_time,
                'completionTimestamp': end_time,
                "steps": _thread_local.buffer
            })
            _thread_local.buffer = []
            del _thread_local.baserun_id
            return result

        return wrapper

    @staticmethod
    def log(message, payload=None):
        if not Baserun._initialized:
            return

        baserun_id = getattr(_thread_local, "baserun_id", None)
        if not baserun_id:
            warnings.warn("baserun.log was called outside of a Baserun decorated test. The log will be ignored.")
            return

        if not hasattr(_thread_local, 'buffer'):
            _thread_local.buffer = []

        log_entry = {
            "message": message,
            "payload": payload
        }
        _thread_local.buffer.append(log_entry)

    @staticmethod
    def store_test(test_data: dict):
        Baserun._tests.append(test_data)

    @staticmethod
    def flush():
        if not Baserun._initialized:
            warnings.warn("Baserun has not been initialized. No data will be flushed.")
            return

        if not Baserun._tests:
            return

        headers = {"Authorization": f"Bearer {Baserun._api_key}"}
        try:
            response = requests.post(Baserun._api_url, json={"tests": Baserun._tests}, headers=headers)
            response.raise_for_status()
        except requests.RequestException as e:
            warnings.warn(f"Failed to upload results to Baserun: {str(e)}")

        Baserun._tests = []
