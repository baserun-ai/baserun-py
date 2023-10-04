import os
from unittest.mock import patch, call

import openai
import pytest
from dotenv import load_dotenv
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from baserun import Baserun
from baserun.v1.baserun_pb2 import Run, Span, StartRunRequest

load_dotenv()


@pytest.fixture(autouse=True)
def set_openai_api_key():
    api_key = os.environ.get("OPENAI_API_KEY")

    openai.api_key = api_key


@pytest.fixture
def mock_services():
    with patch("baserun.grpc.submission_service.StartRun") as mock_start_run, patch(
        "baserun.grpc.submission_service.SubmitLog"
    ) as mock_submit_log, patch(
        "baserun.grpc.submission_service.SubmitSpan"
    ) as mock_submit_span, patch(
        "baserun.grpc.submission_service.EndRun"
    ) as mock_end_run, patch(
        "baserun.grpc.submission_service.SubmitEval"
    ) as mock_submit_eval, patch(
        "baserun.grpc.submission_service.StartTestSuite"
    ) as mock_start_test_suite, patch(
        "baserun.grpc.submission_service.EndTestSuite"
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


def get_mock_objects(mock_services) -> tuple[Run, Span, Run, Run]:
    mock_services["mock_start_run"].assert_called_once()
    run_call: call = mock_services["mock_start_run"].call_args_list[0]
    start_run_request: StartRunRequest = run_call.args[0]
    started_run = start_run_request.run

    if len(mock_services["mock_submit_span"].call_args_list):
        mock_services["mock_submit_span"].assert_called_once()
        submit_span_call: call = mock_services["mock_submit_span"].call_args_list[0]
        submit_span_request = submit_span_call.args[0]
        span = submit_span_request.span
        submitted_run = submit_span_request.run
    else:
        span = None
        submitted_run = None

    mock_services["mock_end_run"].assert_called_once()
    end_run_call: call = mock_services["mock_end_run"].call_args_list[0]
    end_run_request: StartRunRequest = end_run_call.args[0]
    ended_run = end_run_request.run

    return started_run, span, submitted_run, ended_run
