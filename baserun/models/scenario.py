from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field

from baserun.utils import format_string

if TYPE_CHECKING:
    from baserun import evaluate
    from baserun.models.evals import CompletionEval, Eval, TraceEval
    from baserun.models.evaluators import Evaluator
    from baserun.models.experiment import Experiment


class Scenario(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    input: Dict[str, Any]
    experiment: Optional["Experiment"] = Field(exclude=True, default=None)
    actual: Optional[str] = None
    expected: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    contexts: Optional[List[str]] = Field(default_factory=list)

    @classmethod
    def default(cls, experiment: "Experiment") -> "Scenario":
        return cls(
            id=str(uuid4()),
            name=f"{experiment.name} Scenario {len(experiment.scenarios) + 1}",
            input={},
            expected="",
            metadata={},
            experiment=experiment,
        )

    @property
    def client(self):
        return self.experiment.client

    def evaluate(self, evaluators: List["Evaluator"]) -> List[Union["CompletionEval", "TraceEval", "Eval"]]:
        return evaluate(evaluators=evaluators, scenario=self)

    def expected_string(self) -> str:
        return format_string(self._unformatted_expected_string(), **self.input)

    def format_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [{**message, "content": message.get("content", "").format(**self.input)} for message in messages]

    def _unformatted_expected_string(self) -> str:
        if isinstance(self.expected, dict):
            return self.expected.get("content", "")
        if isinstance(self.expected, str):
            return self.expected
        return str(self.expected)

    def model_dump(self, *args, **kwargs):
        dumped = super().model_dump(*args, **kwargs)
        dumped["expected"] = self.expected_string()
        return dumped
