from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import (
    ClassVar as _ClassVar,
    Iterable as _Iterable,
    Mapping as _Mapping,
    Optional as _Optional,
    Union as _Union,
)

DESCRIPTOR: _descriptor.FileDescriptor

class Status(_message.Message):
    __slots__ = ["message", "code"]

    class StatusCode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
        STATUS_CODE_UNSPECIFIED: _ClassVar[Status.StatusCode]
        STATUS_CODE_OK: _ClassVar[Status.StatusCode]
        STATUS_CODE_ERROR: _ClassVar[Status.StatusCode]
    STATUS_CODE_UNSPECIFIED: Status.StatusCode
    STATUS_CODE_OK: Status.StatusCode
    STATUS_CODE_ERROR: Status.StatusCode
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    CODE_FIELD_NUMBER: _ClassVar[int]
    message: str
    code: Status.StatusCode
    def __init__(
        self,
        message: _Optional[str] = ...,
        code: _Optional[_Union[Status.StatusCode, str]] = ...,
    ) -> None: ...

class ToolFunction(_message.Message):
    __slots__ = ["name", "arguments"]
    NAME_FIELD_NUMBER: _ClassVar[int]
    ARGUMENTS_FIELD_NUMBER: _ClassVar[int]
    name: str
    arguments: str
    def __init__(
        self, name: _Optional[str] = ..., arguments: _Optional[str] = ...
    ) -> None: ...

class ToolCall(_message.Message):
    __slots__ = ["id", "type", "function"]
    ID_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    FUNCTION_FIELD_NUMBER: _ClassVar[int]
    id: str
    type: str
    function: ToolFunction
    def __init__(
        self,
        id: _Optional[str] = ...,
        type: _Optional[str] = ...,
        function: _Optional[_Union[ToolFunction, _Mapping]] = ...,
    ) -> None: ...

class Message(_message.Message):
    __slots__ = [
        "role",
        "content",
        "finish_reason",
        "function_call",
        "tool_calls",
        "tool_call_id",
        "name",
        "system_fingerprint",
    ]
    ROLE_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    FINISH_REASON_FIELD_NUMBER: _ClassVar[int]
    FUNCTION_CALL_FIELD_NUMBER: _ClassVar[int]
    TOOL_CALLS_FIELD_NUMBER: _ClassVar[int]
    TOOL_CALL_ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    SYSTEM_FINGERPRINT_FIELD_NUMBER: _ClassVar[int]
    role: str
    content: str
    finish_reason: str
    function_call: str
    tool_calls: _containers.RepeatedCompositeFieldContainer[ToolCall]
    tool_call_id: str
    name: str
    system_fingerprint: str
    def __init__(
        self,
        role: _Optional[str] = ...,
        content: _Optional[str] = ...,
        finish_reason: _Optional[str] = ...,
        function_call: _Optional[str] = ...,
        tool_calls: _Optional[_Iterable[_Union[ToolCall, _Mapping]]] = ...,
        tool_call_id: _Optional[str] = ...,
        name: _Optional[str] = ...,
        system_fingerprint: _Optional[str] = ...,
    ) -> None: ...

class Run(_message.Message):
    __slots__ = [
        "run_id",
        "suite_id",
        "name",
        "inputs",
        "run_type",
        "metadata",
        "start_timestamp",
        "completion_timestamp",
        "result",
        "error",
        "session_id",
        "environment",
    ]

    class RunType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
        RUN_TYPE_TEST: _ClassVar[Run.RunType]
        RUN_TYPE_PRODUCTION: _ClassVar[Run.RunType]
    RUN_TYPE_TEST: Run.RunType
    RUN_TYPE_PRODUCTION: Run.RunType
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    SUITE_ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    INPUTS_FIELD_NUMBER: _ClassVar[int]
    RUN_TYPE_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    START_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    COMPLETION_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    RESULT_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    ENVIRONMENT_FIELD_NUMBER: _ClassVar[int]
    run_id: str
    suite_id: str
    name: str
    inputs: _containers.RepeatedScalarFieldContainer[str]
    run_type: Run.RunType
    metadata: str
    start_timestamp: _timestamp_pb2.Timestamp
    completion_timestamp: _timestamp_pb2.Timestamp
    result: str
    error: str
    session_id: str
    environment: str
    def __init__(
        self,
        run_id: _Optional[str] = ...,
        suite_id: _Optional[str] = ...,
        name: _Optional[str] = ...,
        inputs: _Optional[_Iterable[str]] = ...,
        run_type: _Optional[_Union[Run.RunType, str]] = ...,
        metadata: _Optional[str] = ...,
        start_timestamp: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ...,
        completion_timestamp: _Optional[
            _Union[_timestamp_pb2.Timestamp, _Mapping]
        ] = ...,
        result: _Optional[str] = ...,
        error: _Optional[str] = ...,
        session_id: _Optional[str] = ...,
        environment: _Optional[str] = ...,
    ) -> None: ...

class Log(_message.Message):
    __slots__ = ["run_id", "name", "payload", "timestamp"]
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    run_id: str
    name: str
    payload: str
    timestamp: _timestamp_pb2.Timestamp
    def __init__(
        self,
        run_id: _Optional[str] = ...,
        name: _Optional[str] = ...,
        payload: _Optional[str] = ...,
        timestamp: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ...,
    ) -> None: ...

class EndUser(_message.Message):
    __slots__ = ["id", "identifier"]
    ID_FIELD_NUMBER: _ClassVar[int]
    IDENTIFIER_FIELD_NUMBER: _ClassVar[int]
    id: str
    identifier: str
    def __init__(
        self, id: _Optional[str] = ..., identifier: _Optional[str] = ...
    ) -> None: ...

class Model(_message.Message):
    __slots__ = ["id", "model_name", "provider", "name"]
    ID_FIELD_NUMBER: _ClassVar[int]
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    PROVIDER_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    id: int
    model_name: str
    provider: str
    name: str
    def __init__(
        self,
        id: _Optional[int] = ...,
        model_name: _Optional[str] = ...,
        provider: _Optional[str] = ...,
        name: _Optional[str] = ...,
    ) -> None: ...

class ModelConfig(_message.Message):
    __slots__ = [
        "id",
        "model_id",
        "model",
        "logit_bias",
        "presence_penalty",
        "frequency_penalty",
        "temperature",
        "top_p",
        "top_k",
        "functions",
        "function_call",
    ]
    ID_FIELD_NUMBER: _ClassVar[int]
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    LOGIT_BIAS_FIELD_NUMBER: _ClassVar[int]
    PRESENCE_PENALTY_FIELD_NUMBER: _ClassVar[int]
    FREQUENCY_PENALTY_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    TOP_P_FIELD_NUMBER: _ClassVar[int]
    TOP_K_FIELD_NUMBER: _ClassVar[int]
    FUNCTIONS_FIELD_NUMBER: _ClassVar[int]
    FUNCTION_CALL_FIELD_NUMBER: _ClassVar[int]
    id: int
    model_id: int
    model: Model
    logit_bias: str
    presence_penalty: float
    frequency_penalty: float
    temperature: float
    top_p: float
    top_k: float
    functions: str
    function_call: str
    def __init__(
        self,
        id: _Optional[int] = ...,
        model_id: _Optional[int] = ...,
        model: _Optional[_Union[Model, _Mapping]] = ...,
        logit_bias: _Optional[str] = ...,
        presence_penalty: _Optional[float] = ...,
        frequency_penalty: _Optional[float] = ...,
        temperature: _Optional[float] = ...,
        top_p: _Optional[float] = ...,
        top_k: _Optional[float] = ...,
        functions: _Optional[str] = ...,
        function_call: _Optional[str] = ...,
    ) -> None: ...

class Span(_message.Message):
    __slots__ = [
        "run_id",
        "trace_id",
        "span_id",
        "name",
        "start_time",
        "end_time",
        "status",
        "vendor",
        "request_type",
        "model",
        "total_tokens",
        "completion_tokens",
        "prompt_tokens",
        "prompt_messages",
        "completions",
        "api_base",
        "api_type",
        "functions",
        "function_call",
        "temperature",
        "top_p",
        "n",
        "stream",
        "stop",
        "max_tokens",
        "presence_penalty",
        "frequency_penalty",
        "logit_bias",
        "user",
        "logprobs",
        "echo",
        "suffix",
        "best_of",
        "x_request_id",
        "log_id",
        "top_k",
        "end_user",
        "template_id",
        "template_parameters",
        "template_string",
        "template_version_id",
        "tools",
        "tool_choice",
        "seed",
        "response_format",
        "error_stacktrace",
        "completion_id",
    ]
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    TRACE_ID_FIELD_NUMBER: _ClassVar[int]
    SPAN_ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    START_TIME_FIELD_NUMBER: _ClassVar[int]
    END_TIME_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    VENDOR_FIELD_NUMBER: _ClassVar[int]
    REQUEST_TYPE_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    TOTAL_TOKENS_FIELD_NUMBER: _ClassVar[int]
    COMPLETION_TOKENS_FIELD_NUMBER: _ClassVar[int]
    PROMPT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    PROMPT_MESSAGES_FIELD_NUMBER: _ClassVar[int]
    COMPLETIONS_FIELD_NUMBER: _ClassVar[int]
    API_BASE_FIELD_NUMBER: _ClassVar[int]
    API_TYPE_FIELD_NUMBER: _ClassVar[int]
    FUNCTIONS_FIELD_NUMBER: _ClassVar[int]
    FUNCTION_CALL_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    TOP_P_FIELD_NUMBER: _ClassVar[int]
    N_FIELD_NUMBER: _ClassVar[int]
    STREAM_FIELD_NUMBER: _ClassVar[int]
    STOP_FIELD_NUMBER: _ClassVar[int]
    MAX_TOKENS_FIELD_NUMBER: _ClassVar[int]
    PRESENCE_PENALTY_FIELD_NUMBER: _ClassVar[int]
    FREQUENCY_PENALTY_FIELD_NUMBER: _ClassVar[int]
    LOGIT_BIAS_FIELD_NUMBER: _ClassVar[int]
    USER_FIELD_NUMBER: _ClassVar[int]
    LOGPROBS_FIELD_NUMBER: _ClassVar[int]
    ECHO_FIELD_NUMBER: _ClassVar[int]
    SUFFIX_FIELD_NUMBER: _ClassVar[int]
    BEST_OF_FIELD_NUMBER: _ClassVar[int]
    X_REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    LOG_ID_FIELD_NUMBER: _ClassVar[int]
    TOP_K_FIELD_NUMBER: _ClassVar[int]
    END_USER_FIELD_NUMBER: _ClassVar[int]
    TEMPLATE_ID_FIELD_NUMBER: _ClassVar[int]
    TEMPLATE_PARAMETERS_FIELD_NUMBER: _ClassVar[int]
    TEMPLATE_STRING_FIELD_NUMBER: _ClassVar[int]
    TEMPLATE_VERSION_ID_FIELD_NUMBER: _ClassVar[int]
    TOOLS_FIELD_NUMBER: _ClassVar[int]
    TOOL_CHOICE_FIELD_NUMBER: _ClassVar[int]
    SEED_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_FORMAT_FIELD_NUMBER: _ClassVar[int]
    ERROR_STACKTRACE_FIELD_NUMBER: _ClassVar[int]
    COMPLETION_ID_FIELD_NUMBER: _ClassVar[int]
    run_id: str
    trace_id: bytes
    span_id: int
    name: str
    start_time: _timestamp_pb2.Timestamp
    end_time: _timestamp_pb2.Timestamp
    status: Status
    vendor: str
    request_type: str
    model: str
    total_tokens: int
    completion_tokens: int
    prompt_tokens: int
    prompt_messages: _containers.RepeatedCompositeFieldContainer[Message]
    completions: _containers.RepeatedCompositeFieldContainer[Message]
    api_base: str
    api_type: str
    functions: str
    function_call: str
    temperature: float
    top_p: float
    n: int
    stream: bool
    stop: _containers.RepeatedScalarFieldContainer[str]
    max_tokens: int
    presence_penalty: float
    frequency_penalty: float
    logit_bias: str
    user: str
    logprobs: int
    echo: bool
    suffix: str
    best_of: int
    x_request_id: str
    log_id: str
    top_k: float
    end_user: EndUser
    template_id: str
    template_parameters: str
    template_string: str
    template_version_id: str
    tools: str
    tool_choice: str
    seed: int
    response_format: str
    error_stacktrace: str
    completion_id: str
    def __init__(
        self,
        run_id: _Optional[str] = ...,
        trace_id: _Optional[bytes] = ...,
        span_id: _Optional[int] = ...,
        name: _Optional[str] = ...,
        start_time: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ...,
        end_time: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ...,
        status: _Optional[_Union[Status, _Mapping]] = ...,
        vendor: _Optional[str] = ...,
        request_type: _Optional[str] = ...,
        model: _Optional[str] = ...,
        total_tokens: _Optional[int] = ...,
        completion_tokens: _Optional[int] = ...,
        prompt_tokens: _Optional[int] = ...,
        prompt_messages: _Optional[_Iterable[_Union[Message, _Mapping]]] = ...,
        completions: _Optional[_Iterable[_Union[Message, _Mapping]]] = ...,
        api_base: _Optional[str] = ...,
        api_type: _Optional[str] = ...,
        functions: _Optional[str] = ...,
        function_call: _Optional[str] = ...,
        temperature: _Optional[float] = ...,
        top_p: _Optional[float] = ...,
        n: _Optional[int] = ...,
        stream: bool = ...,
        stop: _Optional[_Iterable[str]] = ...,
        max_tokens: _Optional[int] = ...,
        presence_penalty: _Optional[float] = ...,
        frequency_penalty: _Optional[float] = ...,
        logit_bias: _Optional[str] = ...,
        user: _Optional[str] = ...,
        logprobs: _Optional[int] = ...,
        echo: bool = ...,
        suffix: _Optional[str] = ...,
        best_of: _Optional[int] = ...,
        x_request_id: _Optional[str] = ...,
        log_id: _Optional[str] = ...,
        top_k: _Optional[float] = ...,
        end_user: _Optional[_Union[EndUser, _Mapping]] = ...,
        template_id: _Optional[str] = ...,
        template_parameters: _Optional[str] = ...,
        template_string: _Optional[str] = ...,
        template_version_id: _Optional[str] = ...,
        tools: _Optional[str] = ...,
        tool_choice: _Optional[str] = ...,
        seed: _Optional[int] = ...,
        response_format: _Optional[str] = ...,
        error_stacktrace: _Optional[str] = ...,
        completion_id: _Optional[str] = ...,
    ) -> None: ...

class Eval(_message.Message):
    __slots__ = ["name", "type", "result", "score", "submission", "payload"]
    NAME_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    RESULT_FIELD_NUMBER: _ClassVar[int]
    SCORE_FIELD_NUMBER: _ClassVar[int]
    SUBMISSION_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    name: str
    type: str
    result: str
    score: float
    submission: str
    payload: str
    def __init__(
        self,
        name: _Optional[str] = ...,
        type: _Optional[str] = ...,
        result: _Optional[str] = ...,
        score: _Optional[float] = ...,
        submission: _Optional[str] = ...,
        payload: _Optional[str] = ...,
    ) -> None: ...

class Check(_message.Message):
    __slots__ = ["name", "methodology", "expected", "actual", "score", "metadata"]
    NAME_FIELD_NUMBER: _ClassVar[int]
    METHODOLOGY_FIELD_NUMBER: _ClassVar[int]
    EXPECTED_FIELD_NUMBER: _ClassVar[int]
    ACTUAL_FIELD_NUMBER: _ClassVar[int]
    SCORE_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    name: str
    methodology: str
    expected: str
    actual: str
    score: float
    metadata: str
    def __init__(
        self,
        name: _Optional[str] = ...,
        methodology: _Optional[str] = ...,
        expected: _Optional[str] = ...,
        actual: _Optional[str] = ...,
        score: _Optional[float] = ...,
        metadata: _Optional[str] = ...,
    ) -> None: ...

class Feedback(_message.Message):
    __slots__ = ["name", "score", "metadata", "end_user"]
    NAME_FIELD_NUMBER: _ClassVar[int]
    SCORE_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    END_USER_FIELD_NUMBER: _ClassVar[int]
    name: str
    score: float
    metadata: str
    end_user: EndUser
    def __init__(
        self,
        name: _Optional[str] = ...,
        score: _Optional[float] = ...,
        metadata: _Optional[str] = ...,
        end_user: _Optional[_Union[EndUser, _Mapping]] = ...,
    ) -> None: ...

class CompletionAnnotations(_message.Message):
    __slots__ = ["completion_id", "checks", "logs", "feedback"]
    COMPLETION_ID_FIELD_NUMBER: _ClassVar[int]
    CHECKS_FIELD_NUMBER: _ClassVar[int]
    LOGS_FIELD_NUMBER: _ClassVar[int]
    FEEDBACK_FIELD_NUMBER: _ClassVar[int]
    completion_id: str
    checks: _containers.RepeatedCompositeFieldContainer[Check]
    logs: _containers.RepeatedCompositeFieldContainer[Log]
    feedback: _containers.RepeatedCompositeFieldContainer[Feedback]
    def __init__(
        self,
        completion_id: _Optional[str] = ...,
        checks: _Optional[_Iterable[_Union[Check, _Mapping]]] = ...,
        logs: _Optional[_Iterable[_Union[Log, _Mapping]]] = ...,
        feedback: _Optional[_Iterable[_Union[Feedback, _Mapping]]] = ...,
    ) -> None: ...

class TestSuite(_message.Message):
    __slots__ = ["id", "name", "start_timestamp", "completion_timestamp"]
    ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    START_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    COMPLETION_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    id: str
    name: str
    start_timestamp: _timestamp_pb2.Timestamp
    completion_timestamp: _timestamp_pb2.Timestamp
    def __init__(
        self,
        id: _Optional[str] = ...,
        name: _Optional[str] = ...,
        start_timestamp: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ...,
        completion_timestamp: _Optional[
            _Union[_timestamp_pb2.Timestamp, _Mapping]
        ] = ...,
    ) -> None: ...

class Template(_message.Message):
    __slots__ = ["id", "name", "template_type", "template_versions", "active_version"]

    class TemplateType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
        TEMPLATE_TYPE_UNSPECIFIED: _ClassVar[Template.TemplateType]
        TEMPLATE_TYPE_FORMATTED_STRING: _ClassVar[Template.TemplateType]
        TEMPLATE_TYPE_JINJA2: _ClassVar[Template.TemplateType]
    TEMPLATE_TYPE_UNSPECIFIED: Template.TemplateType
    TEMPLATE_TYPE_FORMATTED_STRING: Template.TemplateType
    TEMPLATE_TYPE_JINJA2: Template.TemplateType
    ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    TEMPLATE_TYPE_FIELD_NUMBER: _ClassVar[int]
    TEMPLATE_VERSIONS_FIELD_NUMBER: _ClassVar[int]
    ACTIVE_VERSION_FIELD_NUMBER: _ClassVar[int]
    id: str
    name: str
    template_type: Template.TemplateType
    template_versions: _containers.RepeatedCompositeFieldContainer[TemplateVersion]
    active_version: TemplateVersion
    def __init__(
        self,
        id: _Optional[str] = ...,
        name: _Optional[str] = ...,
        template_type: _Optional[_Union[Template.TemplateType, str]] = ...,
        template_versions: _Optional[
            _Iterable[_Union[TemplateVersion, _Mapping]]
        ] = ...,
        active_version: _Optional[_Union[TemplateVersion, _Mapping]] = ...,
    ) -> None: ...

class TemplateVersion(_message.Message):
    __slots__ = [
        "id",
        "template",
        "tag",
        "parameter_definition",
        "template_string",
        "template_messages",
    ]
    ID_FIELD_NUMBER: _ClassVar[int]
    TEMPLATE_FIELD_NUMBER: _ClassVar[int]
    TAG_FIELD_NUMBER: _ClassVar[int]
    PARAMETER_DEFINITION_FIELD_NUMBER: _ClassVar[int]
    TEMPLATE_STRING_FIELD_NUMBER: _ClassVar[int]
    TEMPLATE_MESSAGES_FIELD_NUMBER: _ClassVar[int]
    id: str
    template: Template
    tag: str
    parameter_definition: str
    template_string: str
    template_messages: _containers.RepeatedCompositeFieldContainer[TemplateMessage]
    def __init__(
        self,
        id: _Optional[str] = ...,
        template: _Optional[_Union[Template, _Mapping]] = ...,
        tag: _Optional[str] = ...,
        parameter_definition: _Optional[str] = ...,
        template_string: _Optional[str] = ...,
        template_messages: _Optional[
            _Iterable[_Union[TemplateMessage, _Mapping]]
        ] = ...,
    ) -> None: ...

class TemplateMessage(_message.Message):
    __slots__ = ["id", "template_version", "message", "role", "order_index"]
    ID_FIELD_NUMBER: _ClassVar[int]
    TEMPLATE_VERSION_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    ROLE_FIELD_NUMBER: _ClassVar[int]
    ORDER_INDEX_FIELD_NUMBER: _ClassVar[int]
    id: str
    template_version: TemplateVersion
    message: str
    role: str
    order_index: int
    def __init__(
        self,
        id: _Optional[str] = ...,
        template_version: _Optional[_Union[TemplateVersion, _Mapping]] = ...,
        message: _Optional[str] = ...,
        role: _Optional[str] = ...,
        order_index: _Optional[int] = ...,
    ) -> None: ...

class Session(_message.Message):
    __slots__ = [
        "id",
        "identifier",
        "start_timestamp",
        "completion_timestamp",
        "end_user",
    ]
    ID_FIELD_NUMBER: _ClassVar[int]
    IDENTIFIER_FIELD_NUMBER: _ClassVar[int]
    START_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    COMPLETION_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    END_USER_FIELD_NUMBER: _ClassVar[int]
    id: str
    identifier: str
    start_timestamp: _timestamp_pb2.Timestamp
    completion_timestamp: _timestamp_pb2.Timestamp
    end_user: EndUser
    def __init__(
        self,
        id: _Optional[str] = ...,
        identifier: _Optional[str] = ...,
        start_timestamp: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ...,
        completion_timestamp: _Optional[
            _Union[_timestamp_pb2.Timestamp, _Mapping]
        ] = ...,
        end_user: _Optional[_Union[EndUser, _Mapping]] = ...,
    ) -> None: ...

class StartRunRequest(_message.Message):
    __slots__ = ["run"]
    RUN_FIELD_NUMBER: _ClassVar[int]
    run: Run
    def __init__(self, run: _Optional[_Union[Run, _Mapping]] = ...) -> None: ...

class StartRunResponse(_message.Message):
    __slots__ = ["message"]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    message: str
    def __init__(self, message: _Optional[str] = ...) -> None: ...

class SubmitLogRequest(_message.Message):
    __slots__ = ["log", "run"]
    LOG_FIELD_NUMBER: _ClassVar[int]
    RUN_FIELD_NUMBER: _ClassVar[int]
    log: Log
    run: Run
    def __init__(
        self,
        log: _Optional[_Union[Log, _Mapping]] = ...,
        run: _Optional[_Union[Run, _Mapping]] = ...,
    ) -> None: ...

class SubmitLogResponse(_message.Message):
    __slots__ = ["message"]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    message: str
    def __init__(self, message: _Optional[str] = ...) -> None: ...

class SubmitSpanRequest(_message.Message):
    __slots__ = ["span", "run"]
    SPAN_FIELD_NUMBER: _ClassVar[int]
    RUN_FIELD_NUMBER: _ClassVar[int]
    span: Span
    run: Run
    def __init__(
        self,
        span: _Optional[_Union[Span, _Mapping]] = ...,
        run: _Optional[_Union[Run, _Mapping]] = ...,
    ) -> None: ...

class SubmitSpanResponse(_message.Message):
    __slots__ = ["message"]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    message: str
    def __init__(self, message: _Optional[str] = ...) -> None: ...

class EndRunRequest(_message.Message):
    __slots__ = ["run"]
    RUN_FIELD_NUMBER: _ClassVar[int]
    run: Run
    def __init__(self, run: _Optional[_Union[Run, _Mapping]] = ...) -> None: ...

class EndRunResponse(_message.Message):
    __slots__ = ["message"]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    message: str
    def __init__(self, message: _Optional[str] = ...) -> None: ...

class SubmitEvalRequest(_message.Message):
    __slots__ = ["eval", "run"]
    EVAL_FIELD_NUMBER: _ClassVar[int]
    RUN_FIELD_NUMBER: _ClassVar[int]
    eval: Eval
    run: Run
    def __init__(
        self,
        eval: _Optional[_Union[Eval, _Mapping]] = ...,
        run: _Optional[_Union[Run, _Mapping]] = ...,
    ) -> None: ...

class SubmitEvalResponse(_message.Message):
    __slots__ = ["message"]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    message: str
    def __init__(self, message: _Optional[str] = ...) -> None: ...

class StartTestSuiteRequest(_message.Message):
    __slots__ = ["test_suite"]
    TEST_SUITE_FIELD_NUMBER: _ClassVar[int]
    test_suite: TestSuite
    def __init__(
        self, test_suite: _Optional[_Union[TestSuite, _Mapping]] = ...
    ) -> None: ...

class StartTestSuiteResponse(_message.Message):
    __slots__ = ["message"]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    message: str
    def __init__(self, message: _Optional[str] = ...) -> None: ...

class EndTestSuiteRequest(_message.Message):
    __slots__ = ["test_suite"]
    TEST_SUITE_FIELD_NUMBER: _ClassVar[int]
    test_suite: TestSuite
    def __init__(
        self, test_suite: _Optional[_Union[TestSuite, _Mapping]] = ...
    ) -> None: ...

class EndTestSuiteResponse(_message.Message):
    __slots__ = ["message"]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    message: str
    def __init__(self, message: _Optional[str] = ...) -> None: ...

class StartSessionRequest(_message.Message):
    __slots__ = ["session", "run"]
    SESSION_FIELD_NUMBER: _ClassVar[int]
    RUN_FIELD_NUMBER: _ClassVar[int]
    session: Session
    run: Run
    def __init__(
        self,
        session: _Optional[_Union[Session, _Mapping]] = ...,
        run: _Optional[_Union[Run, _Mapping]] = ...,
    ) -> None: ...

class StartSessionResponse(_message.Message):
    __slots__ = ["message", "session"]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    SESSION_FIELD_NUMBER: _ClassVar[int]
    message: str
    session: Session
    def __init__(
        self,
        message: _Optional[str] = ...,
        session: _Optional[_Union[Session, _Mapping]] = ...,
    ) -> None: ...

class EndSessionRequest(_message.Message):
    __slots__ = ["session"]
    SESSION_FIELD_NUMBER: _ClassVar[int]
    session: Session
    def __init__(self, session: _Optional[_Union[Session, _Mapping]] = ...) -> None: ...

class EndSessionResponse(_message.Message):
    __slots__ = ["message", "session"]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    SESSION_FIELD_NUMBER: _ClassVar[int]
    message: str
    session: Session
    def __init__(
        self,
        message: _Optional[str] = ...,
        session: _Optional[_Union[Session, _Mapping]] = ...,
    ) -> None: ...

class SubmitTemplateVersionRequest(_message.Message):
    __slots__ = ["template_version", "environment"]
    TEMPLATE_VERSION_FIELD_NUMBER: _ClassVar[int]
    ENVIRONMENT_FIELD_NUMBER: _ClassVar[int]
    template_version: TemplateVersion
    environment: str
    def __init__(
        self,
        template_version: _Optional[_Union[TemplateVersion, _Mapping]] = ...,
        environment: _Optional[str] = ...,
    ) -> None: ...

class SubmitTemplateVersionResponse(_message.Message):
    __slots__ = ["message", "template_version", "template"]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    TEMPLATE_VERSION_FIELD_NUMBER: _ClassVar[int]
    TEMPLATE_FIELD_NUMBER: _ClassVar[int]
    message: str
    template_version: TemplateVersion
    template: Template
    def __init__(
        self,
        message: _Optional[str] = ...,
        template_version: _Optional[_Union[TemplateVersion, _Mapping]] = ...,
        template: _Optional[_Union[Template, _Mapping]] = ...,
    ) -> None: ...

class SubmitModelConfigRequest(_message.Message):
    __slots__ = ["model_config"]
    MODEL_CONFIG_FIELD_NUMBER: _ClassVar[int]
    model_config: ModelConfig
    def __init__(
        self, model_config: _Optional[_Union[ModelConfig, _Mapping]] = ...
    ) -> None: ...

class SubmitModelConfigResponse(_message.Message):
    __slots__ = ["message", "model_config"]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    MODEL_CONFIG_FIELD_NUMBER: _ClassVar[int]
    message: str
    model_config: ModelConfig
    def __init__(
        self,
        message: _Optional[str] = ...,
        model_config: _Optional[_Union[ModelConfig, _Mapping]] = ...,
    ) -> None: ...

class SubmitUserRequest(_message.Message):
    __slots__ = ["user"]
    USER_FIELD_NUMBER: _ClassVar[int]
    user: EndUser
    def __init__(self, user: _Optional[_Union[EndUser, _Mapping]] = ...) -> None: ...

class SubmitUserResponse(_message.Message):
    __slots__ = ["message", "user"]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    USER_FIELD_NUMBER: _ClassVar[int]
    message: str
    user: EndUser
    def __init__(
        self,
        message: _Optional[str] = ...,
        user: _Optional[_Union[EndUser, _Mapping]] = ...,
    ) -> None: ...

class GetTemplatesRequest(_message.Message):
    __slots__ = ["environment"]
    ENVIRONMENT_FIELD_NUMBER: _ClassVar[int]
    environment: str
    def __init__(self, environment: _Optional[str] = ...) -> None: ...

class GetTemplatesResponse(_message.Message):
    __slots__ = ["templates"]
    TEMPLATES_FIELD_NUMBER: _ClassVar[int]
    templates: _containers.RepeatedCompositeFieldContainer[Template]
    def __init__(
        self, templates: _Optional[_Iterable[_Union[Template, _Mapping]]] = ...
    ) -> None: ...

class SubmitAnnotationsRequest(_message.Message):
    __slots__ = ["annotations", "run"]
    ANNOTATIONS_FIELD_NUMBER: _ClassVar[int]
    RUN_FIELD_NUMBER: _ClassVar[int]
    annotations: CompletionAnnotations
    run: Run
    def __init__(
        self,
        annotations: _Optional[_Union[CompletionAnnotations, _Mapping]] = ...,
        run: _Optional[_Union[Run, _Mapping]] = ...,
    ) -> None: ...

class SubmitAnnotationsResponse(_message.Message):
    __slots__ = ["message"]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    message: str
    def __init__(self, message: _Optional[str] = ...) -> None: ...
