import logging
import os
import sys
import uuid
from datetime import datetime

from baserun import Baserun
from baserun.v1.baserun_pb2 import TestSuite, StartTestSuiteRequest, EndTestSuiteRequest
from .grpc import get_or_create_submission_service

logger = logging.getLogger(__name__)


def pytest_addoption(parser):
    parser.addoption(
        "--baserun", action="store_true", help="Enable baserun functionality"
    )


def pytest_sessionstart(session):
    if session.config.getoption("--baserun"):
        Baserun.init()

        sys.argv[0] = os.path.basename(sys.argv[0])
        suite = TestSuite(id=str(uuid.uuid4()), name=" ".join(sys.argv))
        suite.start_timestamp.FromDatetime(datetime.utcnow())

        session.suite = suite
        Baserun.current_test_suite = suite
        try:
            get_or_create_submission_service().StartTestSuite(
                StartTestSuiteRequest(test_suite=suite)
            )
        except Exception as e:
            logger.warning(f"Failed to start test suite for Baserun, error: {e}")


def pytest_sessionfinish(session):
    if session.config.getoption("--baserun"):
        if hasattr(session, "suite"):
            session.suite.completion_timestamp.FromDatetime(datetime.utcnow())

            try:
                get_or_create_submission_service().EndTestSuite(
                    EndTestSuiteRequest(test_suite=session.suite)
                )
            except Exception as e:
                logger.warning(f"Failed to end test suite for Baserun, error: {e}")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--baserun"):
        for item in items:
            item.obj = Baserun.test(item.obj)


def pytest_terminal_summary(terminalreporter):
    # This will happen if they don't run pytest with `--baserun`
    if not Baserun.current_test_suite:
        return

    # TODO: Support other base URLs?
    run_url = f"https://app.baserun.ai/runs/{Baserun.current_test_suite.id}"
    terminalreporter.write_sep("=", "Baserun summary", blue=True)
    terminalreporter.write_line(f"Test results available at: {run_url}")
