import abc
import json
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Type, Union, overload
from uuid import uuid4

from datasets import Dataset
from openai.types.chat import ChatCompletionMessageToolCall

from baserun.integrations.integration import Integration
from baserun.models.tags import Log, Tag, Transform, Variable

if TYPE_CHECKING:
    from baserun.models.evals import CompletionEval, TraceEval
    from baserun.wrappers.generic import GenericClient, GenericCompletion


class CompletionMixin:
    """
    Provides logic that is common to all wrapped completion objects.
    FYI the reason this isn't a pydantic model is each library's "Completion" type is super different, and
    if you have a "base" completion type you will get hella type conflicts.
    """

    def setup_mixin(
        self,
        client: "ClientMixin",
        tags: Optional[List[Tag]] = None,
        evals: Optional[List["CompletionEval"]] = None,
        completion_id: Optional[str] = None,
        tool_results: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ):
        self.client = client
        self.tags = tags or []
        self.evals = evals or []
        self.completion_id = completion_id or ""
        self.tool_results = tool_results or []

    def _clean_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        return {
            k: v
            for k, v in kwargs.items()
            if k
            not in [
                "name",
                "error",
                "completion_id",
                "trace_id",
                "template",
                "config_params",
                "start_timestamp",
                "end_timestamp",
                "first_token_timestamp",
                "tool_results",
                "tags",
                "evals",
                "input_messages",
            ]
        }

    @abc.abstractmethod
    def genericize(self) -> "GenericCompletion": ...

    def add_to_dataset(self, dataset: Dataset) -> Dataset:
        completion_data = self.genericize().model_dump()
        completion_data.pop("start_timestamp", None)
        completion_data.pop("first_token_timestamp", None)
        completion_data.pop("end_timestamp", None)
        completion_data.pop("usage", None)
        completion_data.pop("tool_results", None)
        completion_data.pop("trace_id", None)
        for choice in completion_data["choices"]:
            choice.pop("finish_reason", None)
            choice.pop("logprobs", None)

        client_data = self.client.genericize().model_dump()
        client_data.pop("start_timestamp", None)
        client_data.pop("end_timestamp", None)
        completion_data["client"] = client_data
        return dataset.add_item({"completion": completion_data})

    def tag(self, key: str, value: str, metadata: Optional[Dict[str, Any]] = None, tag_type: Optional[str] = "custom"):
        new_tag = Tag(
            target_type="completion",
            target_id=self.completion_id,
            tag_type=tag_type,
            key=key,
            value=value,
            metadata=metadata or {},
        )
        self.tags.append(new_tag)
        if self.client.autosubmit:
            self.submit_to_baserun()
        return new_tag

    def eval_many(
        self, score_dict: Dict[str, Iterable[float]], metadata: Optional[Dict[str, Any]] = None
    ) -> List["CompletionEval"]:
        """Submit multiple evals at once from scores in a dictionary. (This is compatible with Ragas)"""
        submitted_evals = []
        for name, scores in score_dict.items():
            for score in scores:
                submitted_evals.append(self.eval(name, score=score, metadata=metadata))

        self.evals.extend(submitted_evals)
        if self.client.autosubmit:
            self.submit_to_baserun()
        # Notably, return only the evals added here, and not _all_ evals
        return submitted_evals

    def eval(
        self,
        name: str,
        score: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "CompletionEval":
        from baserun.models.evals import CompletionEval

        evaluation = CompletionEval(target=self, name=name, metadata=metadata or {}, score=score)
        self.evals.append(evaluation)

        if score is not None and self.client.autosubmit:
            self.submit_to_baserun()
        return evaluation

    def feedback(self, name: str, score: Any, metadata: Optional[Dict[str, Any]] = None):
        value = json.dumps(score)

        new_tag = Tag(
            target_type="completion",
            target_id=self.completion_id,
            tag_type="feedback",
            key=name,
            value=value,
            metadata=metadata or {},
        )
        self.tags.append(new_tag)
        if self.client.autosubmit:
            self.submit_to_baserun()
        return new_tag

    @overload
    def log(self, message: str, name: str):
        """Given message and name"""

    @overload
    def log(self, message: str, name: str, metadata: Dict):
        """Given message, name, and metadata."""
        pass

    @overload
    def log(self, message: str):
        """Given just message"""
        pass

    def log(self, message: str, name: Optional[str] = None, metadata: Optional[Dict] = None):
        # Allow for transposition of name and metadata
        if isinstance(name, dict):
            metadata = name
            name = None
        elif isinstance(metadata, str):
            name = metadata
            metadata = {}

        new_tag = Log(
            target_type="completion",
            target_id=self.completion_id,
            key=name or "log",
            value=message,
            metadata=metadata or {},
        )
        self.tags.append(new_tag)
        if self.client.autosubmit:
            self.submit_to_baserun()
        return new_tag

    def tool_result(self, tool_call: ChatCompletionMessageToolCall, result: Any):
        self.tool_results.append({"tool_call": tool_call.model_dump(), "result": result})
        if self.client.autosubmit:
            self.submit_to_baserun()

    def transform(self, *args, **kwargs):
        self.tags.append(Transform(name=args[0], target_type="trace", target_id=self.completion_id, **kwargs))
        if self.client.autosubmit:
            self.submit_to_baserun()

    def variable(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> Tag:
        new_tag = Variable(
            name=key,
            value=value if isinstance(value, str) else json.dumps(value),
            target_type="completion",
            target_id=self.completion_id,
            metadata=metadata or {},
        )
        self.tags.append(new_tag)
        if self.client.autosubmit:
            self.submit_to_baserun()
        return new_tag

    def submit_to_baserun(self):
        pass


class ClientMixin:
    """
    Provides logic that is common to all wrapped client / trace objects.
    FYI the reason this isn't a pydantic model is each library's "Client" type is super different, and
    if you have a "base" client type you will get hella type conflicts.
    """

    def setup_mixin(
        self,
        trace_id: Optional[str] = None,
        output: Optional[str] = None,
        autosubmit: bool = True,
        tags: Optional[List[Tag]] = None,
        evals: Optional[List["TraceEval"]] = None,
        **kwargs,
    ):
        self.tags: List[Tag] = tags or []
        self.evals: List["TraceEval"] = evals or []
        self.trace_id = trace_id or str(uuid4())
        self.output = output
        self.integrations: List[Integration] = []
        self.autosubmit = autosubmit

    @abc.abstractmethod
    def genericize(self) -> "GenericClient": ...

    def tag(self, key: str, value: str, metadata: Optional[Dict[str, Any]] = None, tag_type: Optional[str] = "custom"):
        new_tag = Tag(
            target_type="trace",
            target_id=self.trace_id,
            tag_type=tag_type,
            key=key,
            value=value,
            metadata=metadata or {},
        )
        self.tags.append(new_tag)
        if self.autosubmit:  # type: ignore[attr-defined]
            self.submit_to_baserun()
        return new_tag

    def eval_many(
        self, score_dict: Dict[str, Iterable[float]], metadata: Optional[Dict[str, Any]] = None
    ) -> List["TraceEval"]:
        """Submit multiple evals at once from scores in a dictionary. (This is compatible with Ragas)"""
        submitted_evals = []
        for name, scores in score_dict.items():
            for score in scores:
                submitted_evals.append(self.eval(name, score=score, metadata=metadata))

        if self.autosubmit:  # type: ignore[attr-defined]
            self.submit_to_baserun()
        # Notably, return only the evals added here, and not _all_ evals
        return submitted_evals

    def eval(self, name: str, score: Optional[float] = None, metadata: Optional[Dict[str, Any]] = None) -> "TraceEval":
        from baserun.models.evals import TraceEval

        evaluation = TraceEval(target=self, name=name, metadata=metadata or {}, score=score)
        self.evals.append(evaluation)

        if score is not None and self.autosubmit:  # type: ignore[attr-defined]
            self.submit_to_baserun()
        return evaluation

    def feedback(self, name: str, score: Any, metadata: Optional[Dict[str, Any]] = None):
        value = json.dumps(score)

        new_tag = Tag(
            target_type="trace",
            target_id=self.trace_id,
            tag_type="feedback",
            key=name,
            value=value,
            metadata=metadata or {},
        )
        self.tags.append(new_tag)
        if self.autosubmit:  # type: ignore[attr-defined]
            self.submit_to_baserun()
        return new_tag

    @overload
    def log(self, message: Union[Dict[str, str], str], name: str):
        """Given message and name"""

    @overload
    def log(self, message: Union[Dict[str, str], str], name: str, metadata: Dict):
        """Given message, name, and metadata."""
        pass

    @overload
    def log(self, message: Union[Dict[str, str], str]):
        """Given just message"""
        pass

    def log(self, message: Union[Dict[str, str], str], name: Optional[str] = None, metadata: Optional[Dict] = None):
        # Allow for transposition of name and metadata
        if isinstance(name, dict):
            metadata = name
            name = None
        elif isinstance(metadata, str):
            name = metadata
            metadata = {}

        message = json.dumps(message) if isinstance(message, dict) else message
        new_tag = Log(
            target_type="trace",
            target_id=self.trace_id,
            key=name or "log",
            value=message,
            metadata=metadata or {},
        )
        self.tags.append(new_tag)
        if self.autosubmit:  # type: ignore[attr-defined]
            self.submit_to_baserun()
        return new_tag

    def integrate(self, integration_class: Type[Integration]):
        integration = integration_class(client=self)
        integration.instrument()
        self.integrations.append(integration)

    def transform(self, *args, **kwargs):
        transform_input = kwargs.pop("input", None)
        transform_output = kwargs.pop("output", None)
        new_tag = Transform(
            key=args[0],
            target_type="trace",
            target_id=self.trace_id,
            input=transform_input if isinstance(transform_input, str) else json.dumps(transform_input),
            output=transform_output if isinstance(transform_output, str) else json.dumps(transform_output),
            **kwargs,
        )
        self.tags.append(new_tag)
        if self.autosubmit:
            self.submit_to_baserun()
        return new_tag

    def variable(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None):
        new_tag = Variable(
            name=key,
            value=value if isinstance(value, str) else json.dumps(value),
            target_type="trace",
            target_id=self.trace_id,
            metadata=metadata or {},
        )
        self.tags.append(new_tag)
        if self.autosubmit:  # type: ignore[attr-defined]
            self.submit_to_baserun()
        return new_tag

    def submit_to_baserun(self):
        pass
