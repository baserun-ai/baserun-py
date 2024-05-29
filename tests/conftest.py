import os
from queue import Empty

import openai
import pytest
from dotenv import load_dotenv

from baserun.api import exporter_queue

load_dotenv()


@pytest.fixture(autouse=True)
def set_openai_api_key():
    api_key = os.environ.get("OPENAI_API_KEY")

    openai.api_key = api_key


def pytest_sessionstart(session):
    os.environ["LOG_LEVEL"] = "DEBUG"
    exporter_queue.put(None)


def pytest_runtest_teardown(item, nextitem):
    pass


def get_queued_objects():
    queued_objects = []
    while not exporter_queue.empty():
        try:
            queued_objects.append(exporter_queue.get_nowait())
        except Empty:
            break
    return queued_objects
