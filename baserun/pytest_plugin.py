import logging

from baserun import api

logger = logging.getLogger(__name__)


def pytest_addoption(parser):
    parser.addoption("--baserun", action="store_true", help="Enable baserun functionality")


def pytest_runtest_teardown(item, nextitem):
    while not api.exporter_queue.empty():
        api.exporter_queue.get_nowait()

    for task in api.tasks:
        task.cancel()
    api.tasks = []


def pytest_sessionstart(session):
    # This will stop start_worker from starting worker threads
    api.exporter_thread = "Dummy"


def pytest_sessionfinish(session, exitstatus):
    api.exporter_queue.close()
