from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from baserun.mixins import ClientMixin, CompletionMixin
from baserun.models.scenario import Scenario

if TYPE_CHECKING:
    pass


class Eval(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    id: str = Field(default_factory=lambda: str(uuid4()))
    target: Optional[
        Union[
            "ClientMixin",
            "CompletionMixin",
        ]
    ] = Field(exclude=True, default=None)
    target_type: Optional[str] = None
    target_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    name: Optional[str] = None
    score: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def model_dump(self, *args, **kwargs):
        dumped = super().model_dump()
        dumped["timestamp"] = dumped.pop("timestamp").isoformat()
        return dumped

    def _format_from_vars(self, string_to_format: str) -> str:
        if not self.target or not self.target.tags:
            return string_to_format

        variables = {tag.key: tag.value for tag in self.target.tags if tag.tag_type == "variable"}
        if variables:
            return string_to_format.format(**variables)

        return string_to_format

    def __bool__(self):
        return self.score == 1


class CompletionEval(Eval):
    def __init__(
        self,
        target: Optional["CompletionMixin"] = None,
        target_id: Optional[str] = None,
        **data,
    ):
        if not target_id and target:
            target_id = target.completion_id

        if not target_id:
            raise ValueError("target or target_id is required")

        super().__init__(
            target_type="completion",
            target_id=target_id,
            **data,
            timestamp=datetime.now(timezone.utc),
            target=target,
        )

    def _message_content(self):
        if hasattr(self.target, "captured_choices") and self.target.captured_choices:
            return "".join([d.delta.content for d in self.target.captured_choices if d.delta.content])
        else:
            return self.target.choices[0].message.content

    def includes(self, expected: str, actual: Optional[str] = None) -> Eval:
        from baserun.models.evaluators import Includes

        if not self.target or not isinstance(self.target, CompletionMixin):
            raise ValueError("CompletionEval target must be a Completion")

        input_vars = (
            {tag.key: tag.value for tag in self.target.tags if tag.tag_type == "variable"} if self.target else {}
        )

        generic = self.target.genericize()
        actual = actual or generic.message_content() if self.target and hasattr(generic, "message_content") else actual

        scenario = Scenario(name=self.name or "Scenario", input=input_vars, expected=expected, actual=actual)

        evaluator = Includes(
            scenario=scenario, name=self.name, client=self.target.client, completion=self.target, expected=expected
        )
        evaluation = evaluator.evaluate(actual=actual)
        self.score = evaluation.score
        self.metadata = evaluation.metadata
        self.target.submit_to_baserun()
        return evaluation


class TraceEval(Eval):
    def __init__(
        self,
        target: Optional["ClientMixin"] = None,
        target_id: Optional[str] = None,
        **data,
    ):
        if target:
            target_id = target_id or target.trace_id

        if not target_id:
            raise ValueError("target or target_id is required")

        super().__init__(
            target_type="trace", target_id=target_id, target=target, **data, timestamp=datetime.now(timezone.utc)
        )

    def includes(self, expected: str, actual: Optional[str] = None) -> Eval:
        from baserun.models.evaluators import Includes

        if not self.target or not isinstance(self.target, ClientMixin):
            raise ValueError("TraceEval target must be a Client")

        input_vars = (
            {tag.key: tag.value for tag in self.target.tags if tag.tag_type == "variable"} if self.target else {}
        )
        actual = actual or self.target.output if self.target and hasattr(self.target, "output") else actual

        scenario = Scenario(name=self.name or "Scenario", input=input_vars, expected=expected, actual=actual)

        evaluator = Includes(name=self.name, scenario=scenario, client=self.target, expected=expected)
        evaluation = evaluator.evaluate(actual=actual)
        self.score = evaluation.score
        self.metadata = evaluation.metadata
        self.target.submit_to_baserun()
        return evaluation
