import abc
import json
from abc import ABC
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, overload

from openai.types.chat import ChatCompletionMessageToolCall

from baserun.integrations.integration import Integration
from baserun.models.evals import CompletionEval, TraceEval
from baserun.models.tags import Log, Tag, Transform, Variable

if TYPE_CHECKING:
    from baserun.wrappers.generic import GenericClient, GenericCompletion


class CompletionMixin(ABC):
    """
    Provides logic that is common to all wrapped completion objects.
    FYI the reason this isn't a pydantic model is each library's "Completion" type is super different, and
    if you have a "base" completion type you will get hella type conflicts.
    """

    tags: List[Tag]
    evals: List[CompletionEval]
    completion_id: str
    tool_results: List[Dict[str, Any]]

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
        self.submit_to_baserun()
        return new_tag

    def eval(self, name: str, score: Optional[int] = None, metadata: Optional[Dict[str, Any]] = None) -> CompletionEval:
        evaluator = CompletionEval(target=self, name=name, metadata=metadata or {}, score=score)
        self.evals.append(evaluator)
        return evaluator

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
        self.submit_to_baserun()

    def tool_result(self, tool_call: ChatCompletionMessageToolCall, result: Any):
        self.tool_results.append({"tool_call": tool_call.model_dump(), "result": result})
        self.submit_to_baserun()

    def transform(self, *args, **kwargs):
        self.tags.append(Transform(name=args[0], target_type="trace", target_id=self.completion_id, **kwargs))
        self.submit_to_baserun()

    def variable(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None):
        self.tags.append(
            Variable(
                name=key,
                value=value if isinstance(value, str) else json.dumps(value),
                target_type="completion",
                target_id=self.completion_id,
                metadata=metadata or {},
            )
        )
        self.submit_to_baserun()

    def submit_to_baserun(self):
        pass


class ClientMixin(ABC):
    """
    Provides logic that is common to all wrapped client / trace objects.
    FYI the reason this isn't a pydantic model is each library's "Client" type is super different, and
    if you have a "base" client type you will get hella type conflicts.
    """

    tags: List[Tag]
    evals: List["TraceEval"]
    trace_id: str
    output: Optional[str]
    integrations: List[Integration]

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
        self.submit_to_baserun()
        return new_tag

    def eval(self, name: str, score: Optional[int] = None, metadata: Optional[Dict[str, Any]] = None) -> "TraceEval":
        evaluator = TraceEval(target=self, name=name, metadata=metadata or {}, score=score)
        self.evals.append(evaluator)
        return evaluator

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
            target_type="trace",
            target_id=self.trace_id,
            key=name or "log",
            value=message,
            metadata=metadata or {},
        )
        self.tags.append(new_tag)
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
        self.submit_to_baserun()
        return new_tag

    def submit_to_baserun(self):
        pass


TraceEval.model_rebuild()
CompletionEval.model_rebuild()
