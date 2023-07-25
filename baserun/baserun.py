import logging
import uuid
import requests
import threading
import contextlib
import warnings
import time

_thread_local = threading.local()


class BaserunHandler(logging.Handler):
    def __init__(self, api_url: str, api_key: str) -> None:
        super().__init__()
        self.api_url = api_url
        self.api_key = api_key
        self.buffer = []

    def flush(self, metadata=None):
        if not self.buffer:
            return
        headers = {"Authorization": f"Bearer {self.api_key}"}
        requests.post(self.api_url, json={"tests": [{**metadata, "steps": self.buffer}]}, headers=headers)
        self.buffer = []

    def emit(self, record: logging.LogRecord) -> None:
        baserun_id = getattr(_thread_local, "baserun_id", None)

        if not baserun_id:
            raise ValueError("Logs meant for Baserun must be inside a Baserun context manager.")

        baserun_payload = getattr(record, 'baserun_payload', {})

        log_entry = {
            "message": record.msg,
            "payload": baserun_payload
        }

        self.buffer.append(log_entry)


class BaserunFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return hasattr(record, 'baserun_payload')


class BaserunTest(contextlib.ContextDecorator):
    def __init__(self, metadata=None):
        self.metadata = dict(metadata or {})

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            if 'name' not in self.metadata:
                self.metadata['name'] = func.__name__
            with self:
                result = func(*args, **kwargs)
                self.metadata['result'] = str(result)
                return result

        return wrapper

    def __enter__(self):
        if not Baserun._initialized:
            raise ValueError("Baserun has not been initialized. Please call baserun.init() first.")

        self.metadata['id'] = str(uuid.uuid4())
        _thread_local.baserun_id = self.metadata['id']
        self.metadata['startTimestamp'] = time.time()

        return self.metadata['id']

    def __exit__(self, *exc):
        self.metadata['completionTimestamp'] = time.time()

        del _thread_local.baserun_id
        logger = logging.getLogger()
        for handler in logger.handlers:
            if isinstance(handler, BaserunHandler):
                handler.flush(self.metadata)


class Baserun:
    _initialized = False

    @staticmethod
    def init(api_key: str, api_url: str = "https://baserun.ai/api/runs") -> None:
        if Baserun._initialized:
            warnings.warn("Baserun has already been initialized. Additional calls to init will be ignored.")
            return

        logger = logging.getLogger()
        handler = BaserunHandler(api_url, api_key)
        handler.setLevel(logging.INFO)

        handler.addFilter(BaserunFilter())
        logger.addHandler(handler)

        Baserun._initialized = True

    @staticmethod
    def test(metadata=None) -> BaserunTest:
        return BaserunTest(metadata)