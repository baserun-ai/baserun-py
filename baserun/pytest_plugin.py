import logging

from baserun import api

logger = logging.getLogger(__name__)


def pytest_addoption(parser):
    parser.addoption("--baserun", action="store_true", help="Enable baserun functionality")


def pytest_runtest_teardown(item, nextitem):
    api.exporter_queue.empty()
    for task in api.tasks:
        task.cancel()
    api.tasks = []


def pytest_sessionstart(session):
    # This will stop start_worker from starting worker threads
    api.exporter_thread = "Dummy"
