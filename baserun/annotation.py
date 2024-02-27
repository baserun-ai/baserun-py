import json
import logging
from contextlib import contextmanager, asynccontextmanager
from typing import Optional, Any, Union

from openai import Stream
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from opentelemetry.sdk.trace import _Span
from opentelemetry.trace import get_current_span, Span

from baserun import Baserun
from baserun.constants import PARENT_SPAN_NAME
from baserun.grpc import (
    get_or_create_submission_service,
    get_or_create_async_submission_service,
)
from baserun.v1.baserun_pb2 import (
    Log,
    Check,
    Feedback,
    Run,
    SubmitAnnotationsRequest,
    CompletionAnnotations,
    InputVariable,
    EndUser,
)

logger = logging.getLogger(__name__)


class Annotation:
    completion_id: str
    span: Span
    input_variables: list[InputVariable]
    logs: list[Log]
    checks: list[Check]
    feedback_list: list[Feedback]

    def __init__(self, completion_id: Optional[str] = None, run: Optional[Run] = None):
        self.run = run or Baserun.get_or_create_current_run()
        self.input_variables = []
        self.logs = []
        self.checks = []
        self.feedback_list = []

        if span := self.try_get_span():
            self.span = span

        if completion_id:
            self.completion_id = completion_id

    @classmethod
    def annotate(cls, completion: Union[None, ChatCompletion, Stream[ChatCompletionChunk]] = None):
        if isinstance(completion, ChatCompletion):
            completion_id = completion.id
            return cls(completion_id=completion_id)
        else:
            return cls()

    @classmethod
    @asynccontextmanager
    async def aanotate(cls, completion: Union[None, ChatCompletion, Stream[ChatCompletionChunk]] = None):
        if not Baserun._initialized:
            yield

        annotation = cls.annotate(completion=completion)
        try:
            yield annotation
        finally:
            try:
                await annotation.asubmit()
            except BaseException as e:
                logger.warning(f"Could not submit annotation to baserun: {e}")

    @classmethod
    @contextmanager
    def with_annotation(cls, completion: Union[None, ChatCompletion, Stream[ChatCompletionChunk]] = None):
        if not Baserun._initialized:
            yield

        annotation = cls.annotate(completion=completion)
        try:
            yield annotation
        finally:
            try:
                annotation.submit()
            except BaseException as e:
                logger.warning(f"Could not submit annotation to baserun: {e}")

    def feedback(
        self,
        name: Optional[str] = None,
        thumbsup: Optional[bool] = None,
        stars: Optional[int] = None,
        score: Optional[float] = None,
        metadata: Optional[dict[str, Any]] = None,
    ):
        if score is None:
            if thumbsup is not None:
                score = 1 if thumbsup else 0
            elif stars is not None:
                score = stars / 5
            else:
                logger.info("Could not calculate feedback score, please pass a score, thumbsup, or stars")
                score = 0.0

        run = Baserun.get_or_create_current_run()
        feedback_kwargs: dict[str, Union[str, int, float, EndUser]] = {
            "name": name or "General Feedback",
            "score": score,
        }
        if metadata:
            feedback_kwargs["metadata"] = json.dumps(metadata)

        if run.session_id and Baserun.sessions:
            end_user = Baserun.sessions.get(run.session_id)
            if end_user:
                feedback_kwargs["end_user"] = end_user

        feedback = Feedback(**feedback_kwargs)  # type: ignore
        self.feedback_list.append(feedback)

    def check(
        self,
        name: str,
        methodology: str,
        expected: dict[str, Any],
        actual: dict[str, Any],
        score: Optional[float] = None,
        metadata: Optional[dict[str, Any]] = None,
    ):
        check = Check(
            name=name,
            methodology=methodology,
            actual=json.dumps(actual),
            expected=json.dumps(expected),
            score=score or 0.0,
            metadata=json.dumps(metadata or {}),
        )
        self.checks.append(check)

    def check_includes(
        self,
        name: str,
        expected: Union[str, list[str]],
        actual: str,
        metadata: Optional[dict[str, Any]] = None,
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

    def input(self, key: str, value: str):
        input_variable = InputVariable(key=key, value=value)
        self.input_variables.append(input_variable)

    def try_get_span(self) -> Union[Span, None]:
        current_span: Union[Span, _Span] = get_current_span()
        if (
            isinstance(current_span, _Span)
            and current_span.is_recording()
            and current_span.name.startswith(f"{PARENT_SPAN_NAME}.")
        ):
            return current_span

        # TODO? Maybe we should create a span or trace
        return None

    def submit(self):
        annotation_message = CompletionAnnotations(
            completion_id=self.completion_id,
            checks=self.checks,
            logs=self.logs,
            feedback=self.feedback_list,
            input_variables=self.input_variables,
        )
        Baserun.add_future(
            get_or_create_submission_service().SubmitAnnotations.future(
                SubmitAnnotationsRequest(annotations=annotation_message, run=self.run)
            )
        )

    async def asubmit(self):
        annotation_message = CompletionAnnotations(
            completion_id=self.completion_id,
            checks=self.checks,
            logs=self.logs,
            feedback=self.feedback_list,
        )

        await get_or_create_async_submission_service().SubmitAnnotations(
            SubmitAnnotationsRequest(annotations=annotation_message, run=self.run)
        )
