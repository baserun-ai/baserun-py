import json
from abc import ABC
from typing import Any, Dict, List, Optional, overload

from openai.types.chat import ChatCompletionMessageToolCall

from baserun.models.evals import CompletionEval, TraceEval
from baserun.models.tags import Log, Tag, Transform, Variable


class CompletionMixin(ABC):
    tags: List[Tag]
    evals: List[CompletionEval]
    completion_id: str
    tool_results: List[Dict[str, Any]]

    # TODO: Should this be an ABC? Or have an ABC somewhere (to define tags, evals, etc without pydantic)
    def annotate(self, key: str, value: str, metadata: Optional[Dict[str, Any]] = None):
        self.tags.append(
            Tag(
                target_type="completion",
                target_id=self.completion_id,
                tag_type="annotation",
                key=key,
                value=value,
                metadata=metadata or {},
            )
        )
        self.submit_to_baserun()

    def eval(self, name: str, score: Optional[int] = None, metadata: Optional[Dict[str, Any]] = None) -> CompletionEval:
        evaluator = CompletionEval(target=self, name=name, metadata=metadata or {}, score=score)
        self.evals.append(evaluator)
        return evaluator

    def feedback(self, name: str, score: Any, metadata: Optional[Dict[str, Any]] = None):
        value = json.dumps(score)
        self.tags.append(
            Tag(
                target_type="completion",
                target_id=self.completion_id,
                tag_type="feedback",
                key=name,
                value=value,
                metadata=metadata or {},
            )
        )
        self.submit_to_baserun()

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

        self.tags.append(
            Log(
                target_type="completion",
                target_id=self.completion_id,
                key=name or "log",
                value=message,
                metadata=metadata or {},
            )
        )
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
    tags: List[Tag]
    evals: List["TraceEval"]
    trace_id: str
    result: Optional[str]

    def annotate(self, key: str, value: str, metadata: Optional[Dict[str, Any]] = None):
        self.tags.append(
            Tag(
                target_type="trace",
                target_id=self.trace_id,
                tag_type="annotation",
                key=key,
                value=value,
                metadata=metadata or {},
            )
        )

    def eval(self, name: str, score: Optional[int] = None, metadata: Optional[Dict[str, Any]] = None) -> "TraceEval":
        evaluator = TraceEval(target=self, name=name, metadata=metadata or {}, score=score)
        self.evals.append(evaluator)
        return evaluator

    def feedback(self, name: str, score: Any, metadata: Optional[Dict[str, Any]] = None):
        value = json.dumps(score)
        self.tags.append(
            Tag(
                target_type="trace",
                target_id=self.trace_id,
                tag_type="feedback",
                key=name,
                value=value,
                metadata=metadata or {},
            )
        )
        self.submit_to_baserun()

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

        self.tags.append(
            Log(
                target_type="trace",
                target_id=self.trace_id,
                key=name or "log",
                value=message,
                metadata=metadata or {},
            )
        )
        self.submit_to_baserun()

    def transform(self, *args, **kwargs):
        transform_input = kwargs.pop("input", None)
        transform_output = kwargs.pop("output", None)
        self.tags.append(
            Transform(
                key=args[0],
                target_type="trace",
                target_id=self.trace_id,
                input=transform_input if isinstance(transform_input, str) else json.dumps(transform_input),
                output=transform_output if isinstance(transform_output, str) else json.dumps(transform_output),
                **kwargs,
            )
        )
        self.submit_to_baserun()

    def variable(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None):
        self.tags.append(
            Variable(
                name=key,
                value=value if isinstance(value, str) else json.dumps(value),
                target_type="trace",
                target_id=self.trace_id,
                metadata=metadata or {},
            )
        )
        self.submit_to_baserun()

    def submit_to_baserun(self):
        pass


TraceEval.model_rebuild()
CompletionEval.model_rebuild()
