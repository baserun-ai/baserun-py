import json
import logging
from contextlib import contextmanager, asynccontextmanager
from numbers import Number
from typing import Any, Union

from openai import Stream
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import get_current_span, Span

from baserun import Baserun
from baserun.constants import PARENT_SPAN_NAME
from baserun.grpc import get_or_create_submission_service, get_or_create_async_submission_service
from baserun.v1.baserun_pb2 import Log, Check, Feedback, SubmitCaptureRequest, CapturedCompletion

logger = logging.getLogger(__name__)


class Capture:
    completion_id: str
    span: Span
    logs: list[Log]
    checks: list[Check]
    feedback_list: list[Feedback]

    def __init__(self, completion_id: str = None):
        self.run = Baserun.get_or_create_current_run()
        self.span = self.try_get_span()
        self.completion_id = completion_id
        self.logs = []
        self.checks = []
        self.feedback_list = []

    @classmethod
    def capture(cls, completion: Union[None, ChatCompletion, Stream[ChatCompletionChunk]] = None):
        completion_id = completion.id if completion else None
        return cls(completion_id=completion_id)

    @classmethod
    @asynccontextmanager
    async def aperform_capture(cls, completion: Union[None, ChatCompletion, Stream[ChatCompletionChunk]] = None):
        if not Baserun._initialized:
            yield

        capture = cls.capture(completion=completion)
        try:
            yield capture
        finally:
            try:
                await capture.asubmit()
            except BaseException as e:
                logger.warning(f"Could not submit capture to baserun: {e}")

    @classmethod
    @contextmanager
    def perform_capture(cls, completion: Union[None, ChatCompletion, Stream[ChatCompletionChunk]] = None):
        if not Baserun._initialized:
            yield

        capture = cls.capture(completion=completion)
        try:
            yield capture
        finally:
            try:
                capture.submit()
            except BaseException as e:
                logger.warning(f"Could not submit capture to baserun: {e}")

    def feedback(
        self, thumbsup: bool = None, stars: Number = None, score: Number = None, metadata: dict[str, Any] = None
    ):
        if score is None:
            if thumbsup is not None:
                score = 1 if thumbsup else 0
            elif stars is not None:
                score = stars / 5
            else:
                logger.info("Could not calculate feedback score, please pass a score, thumbsup, or stars")
                score = 0

        feedback = Feedback(score=score, metadata=metadata)
        self.feedback_list.append(feedback)

    def check(
        self,
        name: str,
        methodology: str,
        expected: dict[str, Any],
        actual: dict[str, Any],
        score: Number = None,
        metadata: dict[str, Any] = None,
    ):
        check = Check(
            name=name,
            methodology=methodology,
            actual=json.dumps(actual),
            expected=json.dumps(expected),
            score=score or 0,
            metadata=json.dumps(metadata or {}),
        )
        self.checks.append(check)

    def check_includes(
        self,
        name: str,
        expected: Union[str, list[str]],
        actual: str,
        metadata: dict[str, Any] = None,
    ):
        expected_list = [expected] if isinstance(expected, str) else expected
        result = any(expected in actual for expected in expected_list)
        return self.check(
            name=name,
            methodology="includes",
            expected={"value": expected},
            actual={"value": actual},
            score=1 if result else 0,
            metadata=metadata,
        )

    def log(self, name: str, metadata: dict[str, Any]):
        log = Log(
            run_id=self.run.run_id,
            name=name,
            payload=json.dumps(metadata),
        )
        self.logs.append(log)

    def try_get_span(self) -> Span:
        current_span: ReadableSpan = get_current_span()
        if current_span and current_span.name.startswith(f"{PARENT_SPAN_NAME}."):
            return current_span

        # TODO? Maybe we should create a span or trace
        return None

    def submit(self):
        capture_message = CapturedCompletion(
            completion_id=self.completion_id, checks=self.checks, logs=self.logs, feedback=self.feedback_list
        )
        get_or_create_submission_service().SubmitCapture.future(
            SubmitCaptureRequest(capture=capture_message, run=self.run)
        )

    async def asubmit(self):
        capture_message = CapturedCompletion(
            completion_id=self.completion_id, checks=self.checks, logs=self.logs, feedback=self.feedback_list
        )

        await get_or_create_async_submission_service().SubmitCapture(
            SubmitCaptureRequest(capture=capture_message, run=self.run)
        )
