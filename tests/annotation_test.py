import json

import pytest

from baserun.annotation import Annotation
from tests.testing_functions import use_annotation


@pytest.mark.asyncio
async def test_annotation(mock_services):
    await use_annotation()

    mock_submit_annotation = mock_services["async_submission_service"].SubmitCapture
    assert mock_submit_annotation.call_count == 1
    args, kwargs = mock_submit_annotation.call_args_list[0]

    submit_annotation_request = args[0]

    assert submit_annotation_request.run.name == "test_annotation"

    annotation: Annotation = submit_annotation_request.annotate

    assert len(annotation.feedback) == 1
    assert annotation.feedback[0].score == 0.4

    assert len(annotation.logs) == 1
    assert "Washington" in annotation.logs[0].payload

    assert len(annotation.checks) == 1
    check = annotation.checks[0]
    assert check.methodology == "includes"
    assert json.loads(check.expected).get("value") == "Washington"
    assert "Washington" in json.loads(check.actual).get("value")
    assert check.score == 1
