import os
from typing import Callable, Dict, Generator, Tuple
from unittest.mock import AsyncMock, Mock, call, create_autospec

import openai
import pytest
from dotenv import load_dotenv

from baserun import Baserun, get_templates
from baserun.grpc import (
    get_or_create_async_submission_service,
    get_or_create_submission_service,
)
from baserun.v1.baserun_pb2 import Run, Span, StartRunRequest

load_dotenv()


@pytest.fixture(autouse=True)
def clear_context():
    Baserun.baserun_contexts = {}


@pytest.fixture(autouse=True)
def set_openai_api_key():
    api_key = os.environ.get("OPENAI_API_KEY")

    openai.api_key = api_key


@pytest.fixture
def mock_services() -> Generator[Dict[str, Mock], None, None]:
    # First, create the services
    get_or_create_async_submission_service()
    get_or_create_submission_service()

    services: Dict[str, Callable] = {
        "submission_service": create_autospec,
        "async_submission_service": AsyncMock,
    }
    rpcs = [
        "EndRun",
        "EndSession",
        "EndTestSuite",
        "GetTemplates",
        "StartRun",
        "StartSession",
        "StartTestSuite",
        "SubmitEval",
        "SubmitLog",
        "SubmitModelConfig",
        "SubmitAnnotations",
        "SubmitSpan",
        "SubmitTemplateVersion",
        "SubmitUser",
        "SubmitInputVariable",
    ]

    # Create a dictionary to hold the mocks
    mock_dict = {}

    # Create mocks for each service
    for service, mocking_fn in services.items():
        original_service = getattr(Baserun, service)
        if mocking_fn == AsyncMock:
            mock_service = mocking_fn(spec=original_service)
        else:
            mock_service = mocking_fn(original_service, instance=True)

        setattr(Baserun, service, mock_service)
        mock_dict[service] = mock_service

        # Mock each RPC method in the service
        for rpc in rpcs:
            rpc_attr = getattr(original_service, rpc, None)

            if mocking_fn == AsyncMock:
                mock_method = mocking_fn(spec=rpc_attr)
                mock_method.future = mocking_fn(spec=rpc_attr)
            else:
                mock_method = mocking_fn(rpc_attr, instance=True)

            setattr(mock_service, rpc, mock_method)

    # Yield the dictionary of mock services
    yield mock_dict

    # Remove the mocked instances so they'll be recreated fresh in the next test
    Baserun.submission_service = None
    Baserun.async_submission_service = None


def pytest_sessionstart(session):
    os.environ["LOG_LEVEL"] = "DEBUG"
    # Baserun.init()
    Baserun.exporter_queue.put(None)


def pytest_runtest_teardown(item, nextitem):
    # Baserun.finish()
    get_templates.clear_cache()
    Baserun.templates = {}


def get_mock_objects(mock_services) -> Tuple[Run, Span, Run, Run]:
    Baserun.finish()

    mock_start_run = mock_services["submission_service"].StartRun.future
    mock_end_run = mock_services["submission_service"].EndRun.future

    mock_start_run.assert_called()
    run_call: call = mock_start_run.call_args_list[-1]
    start_run_request: StartRunRequest = run_call.args[0]
    started_run = start_run_request.run

    queue = list(Baserun.exporter_queue.queue)
    if len(queue):
        assert len(queue) == 1
        span_request = queue[0]
        submitted_run = span_request.run
        span = span_request.span
    else:
        span = None
        submitted_run = None

    mock_end_run.assert_called()
    end_run_call: call = mock_end_run.call_args_list[-1]
    end_run_request: StartRunRequest = end_run_call.args[0]
    ended_run = end_run_request.run

    return started_run, span, submitted_run, ended_run
