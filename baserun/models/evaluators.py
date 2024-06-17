from typing import Any, Dict, List, Optional, Union

from datasets import Dataset
from pydantic import BaseModel, ConfigDict, Field, root_validator

from baserun.mixins import ClientMixin, CompletionMixin  # noqa
from baserun.models.evals import CompletionEval, Eval, TraceEval
from baserun.models.experiment import Experiment  # noqa
from baserun.models.scenario import Scenario
from baserun.wrappers.generic import GenericClient, GenericCompletion


class Evaluator(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    scenario: Scenario
    name: Optional[str] = None
    client: Optional[ClientMixin] = None
    completion: Optional[CompletionMixin] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    score: Optional[float] = None

    @root_validator(pre=True)
    def initialize_fields(cls, values):
        scenario = values.get("scenario")
        client = values.get("client", scenario.experiment.client if scenario and scenario.experiment else None)
        name = values.get("name", cls.__name__)
        metadata = values.get("metadata", {})
        values.update({"client": client, "name": name, "metadata": metadata})

        return values

    @property
    def target(self) -> Union[ClientMixin, CompletionMixin, None]:
        if self.completion:
            return self.completion
        if self.client:
            return self.client

        return None

    def evaluate(self, *args, **kwargs) -> Union[Eval, TraceEval, CompletionEval, None]:
        raise NotImplementedError("Subclasses must implement this method.")

    def _create_completion_eval(
        self, score: Optional[float] = None, metadata: Optional[Dict[str, Any]] = None
    ) -> Union[CompletionEval, None]:
        if metadata and not self.metadata:
            self.metadata = metadata
        if score is not None and not self.score is not None:
            self.score = score

        if self.completion:
            return CompletionEval(target=self.completion, name=self._name, score=self.score, metadata=self.metadata)
        return None

    def _create_trace_eval(
        self, score: Optional[float] = None, metadata: Optional[Dict[str, Any]] = None
    ) -> Union[TraceEval, None]:
        if metadata and not self.metadata:
            self.metadata = metadata
        if score is not None and not self.score is not None:
            self.score = score

        if isinstance(self.client, GenericClient):
            return TraceEval(target=self.client, name=self._name, score=self.score, metadata=self.metadata)
        return None

    def _format_from_vars(self, string_to_format: str) -> str:
        if not self.scenario.input:
            return string_to_format

        return string_to_format.format(**self.scenario.input)

    @property
    def _name(self) -> str:
        raise NotImplementedError("Subclasses must implement this method.")

    def _set_actual(self, actual: Optional[str] = None):
        if self.target and not actual:
            if isinstance(self.target, GenericCompletion):
                actual = self.target.message_content()
            elif isinstance(self.target, GenericClient):
                actual = self.target.output

        self.actual = actual

    def _submit_eval(self):
        evaluation = Eval(
            target_type="unknown",
            target_id="unknown",
            name=self._name,
            score=self.score,
            metadata=self.metadata,
        )

        if self.target:
            if isinstance(self.target, GenericCompletion):
                completion_eval = self._create_completion_eval()
                evaluation = completion_eval if completion_eval else evaluation
                if isinstance(evaluation, CompletionEval):
                    self.target.submit_to_baserun()
            elif isinstance(self.target, GenericClient):
                trace_eval = self._create_trace_eval()
                evaluation = trace_eval if trace_eval else evaluation
                if isinstance(evaluation, TraceEval):
                    self.target.submit_to_baserun()

        return evaluation


class Includes(Evaluator):
    expected: Optional[str] = None
    actual: Optional[str] = None

    @root_validator(pre=True)
    def initialize_fields(cls, values):
        scenario = values.get("scenario")
        expected = values.get("expected", scenario.expected_string() if scenario else None)
        values["expected"] = expected
        return values

    @property
    def _name(self):
        return self.name or "Includes"

    def evaluate(
        self,
        actual: Optional[str] = None,
    ) -> Union[Eval, CompletionEval, TraceEval]:
        actual = actual or self.actual or self.scenario.actual
        if actual:
            self._set_actual(actual or self.actual)

        if not self.expected or not self.actual:
            raise ValueError("Expected and Actual values must be provided.")

        result = self._format_from_vars(self.expected) in self.actual

        if self.metadata is not None:
            self.metadata.update({"expected": self.scenario.expected_string(), "actual": self.actual})

        if self.score is None and result is not None:
            self.score = 1 if result else 0
        if self.target:
            if isinstance(self.target, CompletionMixin):
                completion_eval = self._create_completion_eval()
                if completion_eval is not None:
                    self.target.submit_to_baserun()
                    return completion_eval

            elif isinstance(self.target, ClientMixin):
                trace_eval = self._create_trace_eval()
                if trace_eval:
                    return trace_eval

        return Eval(
            target_type="unknown",
            target_id="unknown",
            name=self._name,
            score=self.score,
            metadata=self.metadata or {},
        )


class Correctness(Evaluator):
    threshold: Optional[float] = Field(default=0.5)
    actual: Optional[str] = None
    question: Optional[str] = None
    contexts: Optional[List[str]] = Field(default_factory=list)
    ground_truth: Optional[str] = None

    @root_validator(pre=True)
    def initialize_fields(cls, values):
        scenario = values.get("scenario")
        contexts = values.get("contexts", scenario.contexts if scenario else [])
        ground_truth = values.get(
            "ground_truth", values.get("expected", scenario.expected_string()) if scenario else None
        )

        values["contexts"] = contexts
        values["ground_truth"] = ground_truth
        values["question"] = values.get("question", scenario.input.get("question") if scenario else None)

        return values

    def evaluate(
        self,
        actual: Optional[str] = None,
        question: Optional[str] = None,
        answer: Optional[str] = None,
    ) -> Union[Eval, None]:
        self._set_actual(answer or actual or self.actual)
        self.question = self._format_from_vars(question or self.question or "")

        if not self.client or self.actual is None:
            raise ValueError("Client and Actual / Answer values must be provided.")

        result = self._model_grade()

        if self.metadata is not None:
            self.metadata.update({"threshold": self.threshold, "actual": self.actual})

        if self.score is None and result is not None:
            self.score = 1 if result else 0

        return self._submit_eval()

    def _model_grade(self) -> Union[List[TraceEval], List[CompletionEval]]:
        return self.ragas_evaluate()

    @property
    def _name(self):
        return self.name or "Correctness"

    def ragas_evaluate(
        self,
        question: Optional[str] = None,
        answer: Optional[str] = None,
        contexts: Optional[List[str]] = None,
        ground_truth: Optional[str] = None,
        client: Optional[GenericClient] = None,
        actual: Optional[str] = None,
    ) -> Union[List[TraceEval], List[CompletionEval]]:
        try:
            from ragas import evaluate
            from ragas.metrics import answer_correctness
        except ImportError:
            # TODO: Implement non-ragas evaluation
            raise ImportError("Ragas must be installed to use this method.")

        self._set_actual(answer or actual or self.actual)
        self.client = client or self.client
        self.contexts = contexts or self.contexts
        self.ground_truth = ground_truth or self.ground_truth

        self.question = question or self.question
        if not self.question and isinstance(self.target, GenericCompletion):
            self.question = self.target.input_messages[-1].content

        if self.question and self.actual and self.contexts and self.ground_truth:
            dataset = Dataset.from_list(
                [
                    {
                        "question": self.question,
                        "answer": self.actual,
                        "contexts": self.contexts,
                        "ground_truth": self.ground_truth,
                    }
                ]
            )
        else:
            raise ValueError("Dataset must have columns 'question', 'actual', 'contexts', and 'ground_truth'.")

        try:
            score = evaluate(dataset, metrics=[answer_correctness])
        except (ValueError, AttributeError):
            raise ValueError("Dataset must have columns 'question', 'answer', 'contexts', and 'ground_truth'.")

        if self.target:
            scores = score.scores.data.to_pydict().get("answer_correctness")
            if isinstance(self.target, GenericCompletion):
                return [
                    CompletionEval(
                        target=self.target,
                        name=self._name,
                        score=scores[0],
                        metadata={
                            "question": self.question,
                            "answer": self.actual,
                            "contexts": self.contexts,
                            "ground_truth": self.ground_truth,
                        },
                    )
                ]
            elif self.client:
                return [
                    TraceEval(
                        target=self.client,
                        name=self._name,
                        score=scores[0],
                        metadata={
                            "question": self.question,
                            "answer": self.actual,
                            "contexts": self.contexts,
                            "ground_truth": self.ground_truth,
                        },
                    )
                ]

        ret: List[TraceEval] = []
        return ret


Eval.model_rebuild()
