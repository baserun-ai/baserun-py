import os
from unittest.mock import patch

import openai
import pytest
from dotenv import load_dotenv
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from baserun import Baserun

load_dotenv()


@pytest.fixture(autouse=True)
def set_openai_api_key():
    api_key = os.environ.get("OPENAI_API_KEY")

    openai.api_key = api_key


@pytest.fixture
def mock_services():
    with patch("baserun.Baserun.submission_service.StartRun") as mock_start_run, patch(
        "baserun.Baserun.submission_service.SubmitLog"
    ) as mock_submit_log, patch(
        "baserun.Baserun.submission_service.SubmitSpan"
    ) as mock_submit_span, patch(
        "baserun.Baserun.submission_service.EndRun"
    ) as mock_end_run, patch(
        "baserun.Baserun.submission_service.SubmitEval"
    ) as mock_submit_eval, patch(
        "baserun.Baserun.submission_service.StartTestSuite"
    ) as mock_start_test_suite, patch(
        "baserun.Baserun.submission_service.EndTestSuite"
    ) as mock_end_test_suite:
        yield {
            "mock_start_run": mock_start_run,
            "mock_submit_log": mock_submit_log,
            "mock_submit_span": mock_submit_span,
            "mock_end_run": mock_end_run,
            "mock_submit_eval": mock_submit_eval,
            "mock_start_test_suite": mock_start_test_suite,
            "mock_end_test_suite": mock_end_test_suite,
        }


def pytest_sessionstart(session):
    """Starting up Baserun in tests requires that these things happen in a specific order:
    - `init`, specifically setting up gRPC
    - Mock services, to replace the services that were just set up
    - Instrument
    - Close channel, simply to ensure that no unmocked calls get through
    """
    Baserun.init(instrument=False)
    # mock_services()
    # Replace the batch processor so that things happen synchronously and not in a separate thread
    Baserun.instrument(processor_class=SimpleSpanProcessor)
    Baserun.grpc_channel.close()
