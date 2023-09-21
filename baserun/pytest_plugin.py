import logging
import time
import uuid

from baserun import Baserun
from baserun.v1.baserun_pb2 import TestSuite, StartTestSuiteRequest, EndTestSuiteRequest

logger = logging.getLogger(__name__)

run_url = None


def pytest_addoption(parser):
    parser.addoption(
        "--baserun", action="store_true", help="Enable baserun functionality"
    )
    parser.addoption(
        "--baserun-api-url", default="https://baserun.ai/api/v1", help="Baserun API URL"
    )
    parser.addoption(
        "--no-flush",
        action="store_true",
        default=False,
        help="do not flush to baserun, even if --baserun is used",
    )


def pytest_sessionstart(session):
    if session.config.getoption("--baserun"):
        api_url = session.config.getoption("--baserun-api-url")
        Baserun.init(api_url)

        suite = TestSuite(id=str(uuid.uuid4()))
        suite.start_timestamp.FromSeconds(int(time.time()))
        session.suite = suite
        Baserun.current_test_suite = suite
        try:
            Baserun.submission_service.StartTestSuite(
                StartTestSuiteRequest(test_suite=suite)
            )
        except Exception as e:
            logger.warning(f"Failed to start test suite for Baserun, error: {e}")


def pytest_sessionfinish(session):
    global run_url
    if session.config.getoption("--baserun"):
        if hasattr(session, "suite"):
            session.suite.completion_timestamp.FromSeconds(int(time.time()))

            try:
                Baserun.submission_service.EndTestSuite(
                    EndTestSuiteRequest(test_suite=session.suite)
                )
            except Exception as e:
                logger.warning(f"Failed to end test suite for Baserun, error: {e}")

        if session.config.getoption("--no-flush"):
            logger.info("Baserun flush disabled by --no-flush option.")
            return

        run_url = Baserun.flush()


def pytest_collection_modifyitems(config, items):
    if config.getoption("--baserun"):
        for item in items:
            item.obj = Baserun.test(item.obj)


def pytest_terminal_summary(terminalreporter):
    global run_url
    if run_url:
        terminalreporter.write_sep("=", "Baserun summary", blue=True)
        terminalreporter.write_line(f"Test results available at: {run_url}")
