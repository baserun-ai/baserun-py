from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from baserun.mixins import ClientMixin, CompletionMixin


class CompletionEval(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    id: str = Field(default_factory=lambda: str(uuid4()))
    target: "CompletionMixin" = Field(exclude=True)
    target_type: str
    target_id: str
    timestamp: datetime
    name: Optional[str] = None
    score: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def __init__(self, target: "CompletionMixin", **data):
        super().__init__(
            target_type="completion",
            target_id=target.completion_id,
            **data,
            timestamp=datetime.now(timezone.utc),
            target=target,
        )

    def __bool__(self):
        return self.score == 1

    def model_dump(self, *args, **kwargs):
        dumped = super().model_dump()
        dumped["timestamp"] = dumped.pop("timestamp").isoformat()
        return dumped

    def not_includes(self, expected: str, actual: Optional[str] = None) -> bool:
        expected = self._format_from_vars(expected)
        actual = actual or self.message_content()
        result = expected in actual
        self.metadata.update({"expected": expected, "actual": actual})
        self.score = 1 if result else 0
        self.target.submit_to_baserun()
        return result

    def includes(self, expected: str, actual: Optional[str] = None) -> bool:
        expected = self._format_from_vars(expected)
        actual = actual or self.message_content()
        result = expected in actual
        self.metadata.update({"expected": expected, "actual": actual})
        self.score = 1 if result else 0
        self.target.submit_to_baserun()
        return result

    def message_content(self):
        if hasattr(self.target, "captured_choices") and self.target.captured_choices:
            return "".join([d.delta.content for d in self.target.captured_choices if d.delta.content])
        else:
            return self.target.choices[0].message.content

    def _format_from_vars(self, string_to_format: str) -> str:
        variables = {tag.key: tag.value for tag in self.target.tags if tag.tag_type == "variable"}
        if variables:
            return string_to_format.format(**variables)

        return string_to_format


class TraceEval(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    id: str = Field(default_factory=lambda: str(uuid4()))
    target: "ClientMixin" = Field(exclude=True)
    target_type: str
    target_id: str
    timestamp: datetime
    name: Optional[str] = None
    score: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def __init__(self, target: "ClientMixin", **data):
        super().__init__(
            target_type="trace", target_id=target.trace_id, target=target, **data, timestamp=datetime.now(timezone.utc)
        )

    def __bool__(self):
        return self.score == 1

    def model_dump(self, *args, **kwargs):
        dumped = super().model_dump()
        dumped["timestamp"] = dumped.pop("timestamp").isoformat()
        return dumped

    def not_includes(self, expected: str, actual: Optional[str] = None) -> bool:
        if not actual:
            actual = self.target.output or ""

        expected = self._format_from_vars(expected)
        result = actual not in expected
        self.metadata.update({"expected": expected, "actual": actual})
        self.score = 1 if result else 0
        self.target.submit_to_baserun()
        return result

    def includes(self, expected: str, actual: Optional[str] = None) -> bool:
        expected = self._format_from_vars(expected)
        result = expected in (actual or self.target.output or "")
        self.metadata.update({"expected": expected, "actual": actual})
        self.score = 1 if result else 0
        self.target.submit_to_baserun()
        return result

    def _format_from_vars(self, string_to_format: str) -> str:
        variables = {tag.key: tag.value for tag in self.target.tags if tag.tag_type == "variable"}
        if variables:
            return string_to_format.format(**variables)

        return string_to_format
