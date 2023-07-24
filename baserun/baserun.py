import logging
import uuid
import requests
import threading
import contextlib

_thread_local = threading.local()


class BaserunHandler(logging.Handler):
    def __init__(self, api_url: str, api_key: str) -> None:
        super().__init__()
        self.api_url = api_url
        self.api_key = api_key
        self.buffer = []

    def flush(self):
        if not self.buffer:
            return
        headers = {"Authorization": f"Bearer {self.api_key}"}
        requests.post(self.api_url, json=self.buffer, headers=headers)
        self.buffer = []

    def emit(self, record: logging.LogRecord) -> None:
        baserun_id = getattr(_thread_local, "baserun_id", None)

        if not baserun_id:
            raise ValueError("Logs meant for Baserun must be inside a Baserun context manager.")

        baserun_payload = getattr(record, 'baserun_payload', {})

        log_entry = {
            "baserun_id": baserun_id,
            "message": record.msg,
            "payload": baserun_payload
        }

        self.buffer.append(log_entry)


class BaserunFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return hasattr(record, 'baserun_payload')


class Baserun:
    @staticmethod
    def init(api_key: str, api_url: str = "https://baserun.ai/api/logs") -> None:
        logger = logging.getLogger()
        handler = BaserunHandler(api_url, api_key)
        handler.setLevel(logging.INFO)

        handler.addFilter(BaserunFilter())
        logger.addHandler(handler)

    @staticmethod
    @contextlib.contextmanager
    def test() -> None:
        baserun_id = str(uuid.uuid4())
        _thread_local.baserun_id = baserun_id
        try:
            yield baserun_id
        finally:
            del _thread_local.baserun_id

            logger = logging.getLogger()
            for handler in logger.handlers:
                if isinstance(handler, BaserunHandler):
                    handler.flush()

