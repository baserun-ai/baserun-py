import uuid  # noqa
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from baserun.api import ApiClient
from baserun.integrations.integration import Integration
from baserun.mixins import ClientMixin, CompletionMixin
from baserun.models.evals import CompletionEval, TraceEval
from baserun.models.experiment import Experiment
from baserun.models.tags import Tag

Model = TypeVar("Model", bound="BaseModel")


class GenericUsage(BaseModel):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int


class GenericInputMessage(BaseModel):
    role: str
    content: Optional[str] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class GenericCompletionMessage(BaseModel):
    role: str = "assistant"
    content: Optional[str] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class GenericChoice(BaseModel):
    index: int = 0
    message: GenericCompletionMessage
    finish_reason: Optional[str] = "stop"
    logprobs: Optional[str] = None


class GenericCompletion(CompletionMixin, BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client: "GenericClient"
    name: str
    error: Optional[str] = None
    model: Optional[str] = None
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    completion_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    template: Optional[str] = None
    start_timestamp: Optional[datetime] = None
    first_token_timestamp: Optional[datetime] = None
    end_timestamp: Optional[datetime] = None
    usage: Optional[GenericUsage] = None
    input_messages: List[GenericInputMessage] = Field(default_factory=list)
    choices: List[GenericChoice] = Field(default_factory=list)
    config_params: Dict[str, Any] = Field(default_factory=dict)
    tool_results: List[Any] = Field(default_factory=list)
    tags: List[Tag] = Field(default_factory=list)
    evals: List[CompletionEval] = Field(default_factory=list)
    request_id: Optional[str] = None
    environment: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def __init__(self, **data):
        client = data.pop("client")
        tags = data.pop("tags", [])
        evals = data.pop("evals", [])
        tool_results = data.pop("tool_results", [])
        super().__init__(client=client, **data)
        self.setup_mixin(client=client, tags=tags, evals=evals, tool_results=tool_results, **data)

    def genericize(self):
        return self

    def model_dump(self, *args, **kwargs):
        dumped = super().model_dump()
        dumped["start_timestamp"] = dumped.pop("start_timestamp").isoformat()
        if dumped["first_token_timestamp"]:
            dumped["first_token_timestamp"] = dumped.pop("first_token_timestamp").isoformat()
        if dumped["end_timestamp"]:
            dumped["end_timestamp"] = dumped.pop("end_timestamp").isoformat()
        return dumped

    def submit_to_baserun(self):
        self.client.api_client.submit_completion(self)

    @property
    def messages(self):
        return [c.message for c in self.choices]

    @messages.setter
    def messages(self, value):
        self.choices = [GenericChoice(message=m, index=i) for m, i in enumerate(value)]
        self.submit_to_baserun()

    def message_content(self):
        return "".join([m.content for m in self.messages if m.content])


class GenericClient(ClientMixin, BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    tags: List[Tag] = Field(default_factory=list)
    evals: List[TraceEval] = Field(default_factory=list)
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    output: Optional[str] = None
    error: Optional[str] = None
    user: Optional[str] = None
    session: Optional[str] = None
    start_timestamp: Optional[datetime] = Field(default_factory=lambda: datetime.now(timezone.utc))
    end_timestamp: Optional[datetime] = None
    api_client: Optional[ApiClient] = Field(default_factory=ApiClient)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    integrations: List[Integration] = Field(default_factory=list)
    autosubmit: bool = Field(default=True)
    experiment: Optional[Experiment] = None

    def __init__(self, **data):
        super().__init__(**data)
        self.setup_mixin(**data)

    def genericize(self):
        return self

    def model_dump(self, *args, **kwargs):
        dumped = super().model_dump()
        dumped.pop("api_client", None)
        dumped.pop("integrations", None)
        dumped["start_timestamp"] = dumped.pop("start_timestamp").isoformat()
        if dumped["end_timestamp"]:
            dumped["end_timestamp"] = dumped.pop("end_timestamp").isoformat()
        return dumped

    def submit_to_baserun(self):
        self.api_client.submit_trace(self)
