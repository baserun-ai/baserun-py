import json

import pytest

from baserun.v1.baserun_pb2 import SubmitAnnotationsRequest
from tests.testing_functions import use_annotation


@pytest.mark.asyncio
async def test_annotation(mock_services):
    await use_annotation()

    mock_submit_annotation = mock_services["async_submission_service"].SubmitAnnotations
    assert mock_submit_annotation.call_count == 1
    args, kwargs = mock_submit_annotation.call_args_list[0]

    submit_annotation_request: SubmitAnnotationsRequest = args[0]

    assert submit_annotation_request.run.name == "test_annotation"

    annotations = submit_annotation_request.annotations

    assert len(annotations.feedback) == 1
    assert annotations.feedback[0].score == 0.8

    assert len(annotations.logs) == 1
    assert "Washington" in annotations.logs[0].payload

    assert len(annotations.checks) == 1
    check = annotations.checks[0]
    assert check.methodology == "includes"
    assert json.loads(check.expected).get("value") == "Washington"
    assert "Washington" in json.loads(check.actual).get("value")
    assert check.score == 1
