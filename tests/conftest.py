import os
from unittest.mock import patch

import openai
import pytest
from dotenv import load_dotenv

from baserun import Baserun

load_dotenv()


@pytest.fixture(autouse=True)
def set_openai_api_key():
    api_key = os.environ.get("OPENAI_API_KEY")

    openai.api_key = api_key


@pytest.fixture(autouse=True)
def mock_services(init_baserun):
    with patch.object(
        Baserun.submission_service, "StartRun", autospec=True
    ) as mock_start_run, patch.object(
        Baserun.submission_service, "SubmitLog", autospec=True
    ) as mock_submit_log, patch.object(
        Baserun.submission_service, "SubmitSpan", autospec=True
    ) as mock_submit_span, patch.object(
        Baserun.submission_service, "EndRun", autospec=True
    ) as mock_end_run, patch.object(
        Baserun.submission_service, "SubmitEval", autospec=True
    ) as mock_submit_eval, patch.object(
        Baserun.submission_service, "StartTestSuite", autospec=True
    ) as mock_start_test_suite, patch.object(
        Baserun.submission_service, "EndTestSuite", autospec=True
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


@pytest.fixture
def init_baserun():
    Baserun.init()
