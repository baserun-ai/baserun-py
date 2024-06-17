import logging
from multiprocessing import Process

from baserun import worker

logger = logging.getLogger(__name__)


def pytest_addoption(parser):
    parser.addoption("--baserun", action="store_true", help="Enable baserun functionality")


def pytest_runtest_teardown(item, nextitem):
    while not worker.exporter_queue.empty():
        worker.exporter_queue.get_nowait()

    for task in worker.tasks:
        task.cancel()
    worker.tasks = []


def pytest_runtest_setup(item):
    while not worker.exporter_queue.empty():
        worker.exporter_queue.get_nowait()

    for task in worker.tasks:
        task.cancel()
    worker.tasks = []


def pytest_sessionstart(session):
    # This will stop start_worker from starting worker processes
    worker.exporter_process = Process()


def pytest_sessionfinish(session, exitstatus):
    worker.exporter_queue.close()
