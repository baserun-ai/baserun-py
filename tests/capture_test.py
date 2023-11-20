import json

import pytest

from baserun.capture import Capture
from tests.testing_functions import use_capture


@pytest.mark.asyncio
async def test_capture(mock_services):
    await use_capture()

    mock_submit_capture = mock_services["async_submission_service"].SubmitCapture
    assert mock_submit_capture.call_count == 1
    args, kwargs = mock_submit_capture.call_args_list[0]

    submit_capture_request = args[0]

    assert submit_capture_request.run.name == "test_capture"

    capture: Capture = submit_capture_request.capture

    assert len(capture.feedback) == 1
    assert capture.feedback[0].score == 0.4

    assert len(capture.logs) == 1
    assert "Washington" in capture.logs[0].payload

    assert len(capture.checks) == 1
    check = capture.checks[0]
    assert check.methodology == "includes"
    assert json.loads(check.expected).get("value") == "Washington"
    assert "Washington" in json.loads(check.actual).get("value")
    assert check.score == 1
