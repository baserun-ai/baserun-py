import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from baserun.api import ApiClient
from baserun.integrations.integration import Integration
from baserun.mixins import ClientMixin, CompletionMixin
from baserun.models.evals import CompletionEval, TraceEval
from baserun.models.tags import Tag

Model = TypeVar("Model", bound="BaseModel")


class GenericUsage(BaseModel):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int


class GenericInputMessage(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class GenericCompletionMessage(BaseModel):
    role: str = "assistant"
    content: Optional[str] = None
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
    trace_id: str
    completion_id: str
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

    def genericize(self):
        return self

    def submit_to_baserun(self):
        self.client.api_client.submit_completion(self)


class GenericClient(ClientMixin, BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    tags: List[Tag] = Field(default_factory=list)
    evals: List[TraceEval] = Field(default_factory=list)
    trace_id: str
    _output: Optional[str] = None
    error: Optional[str] = None
    user: Optional[str] = None
    session: Optional[str] = None
    start_timestamp: Optional[datetime] = None
    end_timestamp: Optional[datetime] = None
    api_client: Optional[ApiClient] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    integrations: List[Integration] = Field(default_factory=list)

    @property
    def output(self):
        return self._output

    @output.setter
    def output(self, value):
        self._output = value
        self.submit_to_baserun()

    def genericize(self):
        return self

    def submit_to_baserun(self):
        self.api_client.submit_trace(self)
