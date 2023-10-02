from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

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
    def __init__(self, message: _Optional[str] = ..., code: _Optional[_Union[Status.StatusCode, str]] = ...) -> None: ...

class Message(_message.Message):
    __slots__ = ["role", "content", "finish_reason", "function_call"]
    ROLE_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    FINISH_REASON_FIELD_NUMBER: _ClassVar[int]
    FUNCTION_CALL_FIELD_NUMBER: _ClassVar[int]
    role: str
    content: str
    finish_reason: str
    function_call: str
    def __init__(self, role: _Optional[str] = ..., content: _Optional[str] = ..., finish_reason: _Optional[str] = ..., function_call: _Optional[str] = ...) -> None: ...

class Run(_message.Message):
    __slots__ = ["run_id", "suite_id", "name", "inputs", "run_type", "metadata", "start_timestamp", "completion_timestamp", "result", "error"]
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
    def __init__(self, run_id: _Optional[str] = ..., suite_id: _Optional[str] = ..., name: _Optional[str] = ..., inputs: _Optional[_Iterable[str]] = ..., run_type: _Optional[_Union[Run.RunType, str]] = ..., metadata: _Optional[str] = ..., start_timestamp: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., completion_timestamp: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., result: _Optional[str] = ..., error: _Optional[str] = ...) -> None: ...

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
    def __init__(self, run_id: _Optional[str] = ..., name: _Optional[str] = ..., payload: _Optional[str] = ..., timestamp: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class Span(_message.Message):
    __slots__ = ["run_id", "trace_id", "span_id", "name", "start_time", "end_time", "status", "vendor", "request_type", "model", "total_tokens", "completion_tokens", "prompt_tokens", "prompt_messages", "completions", "api_base", "api_type", "functions", "function_call", "temperature", "top_p", "n", "stream", "stop", "max_tokens", "presence_penalty", "frequency_penalty", "logit_bias", "user", "logprobs", "echo", "suffix", "best_of", "log_id", "top_k"]
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
    LOG_ID_FIELD_NUMBER: _ClassVar[int]
    TOP_K_FIELD_NUMBER: _ClassVar[int]
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
    log_id: str
    top_k: float
    def __init__(self, run_id: _Optional[str] = ..., trace_id: _Optional[bytes] = ..., span_id: _Optional[int] = ..., name: _Optional[str] = ..., start_time: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., end_time: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., status: _Optional[_Union[Status, _Mapping]] = ..., vendor: _Optional[str] = ..., request_type: _Optional[str] = ..., model: _Optional[str] = ..., total_tokens: _Optional[int] = ..., completion_tokens: _Optional[int] = ..., prompt_tokens: _Optional[int] = ..., prompt_messages: _Optional[_Iterable[_Union[Message, _Mapping]]] = ..., completions: _Optional[_Iterable[_Union[Message, _Mapping]]] = ..., api_base: _Optional[str] = ..., api_type: _Optional[str] = ..., functions: _Optional[str] = ..., function_call: _Optional[str] = ..., temperature: _Optional[float] = ..., top_p: _Optional[float] = ..., n: _Optional[int] = ..., stream: bool = ..., stop: _Optional[_Iterable[str]] = ..., max_tokens: _Optional[int] = ..., presence_penalty: _Optional[float] = ..., frequency_penalty: _Optional[float] = ..., logit_bias: _Optional[str] = ..., user: _Optional[str] = ..., logprobs: _Optional[int] = ..., echo: bool = ..., suffix: _Optional[str] = ..., best_of: _Optional[int] = ..., log_id: _Optional[str] = ..., top_k: _Optional[float] = ...) -> None: ...

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
    def __init__(self, name: _Optional[str] = ..., type: _Optional[str] = ..., result: _Optional[str] = ..., score: _Optional[float] = ..., submission: _Optional[str] = ..., payload: _Optional[str] = ...) -> None: ...

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
    def __init__(self, id: _Optional[str] = ..., name: _Optional[str] = ..., start_timestamp: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., completion_timestamp: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

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
    def __init__(self, log: _Optional[_Union[Log, _Mapping]] = ..., run: _Optional[_Union[Run, _Mapping]] = ...) -> None: ...

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
    def __init__(self, span: _Optional[_Union[Span, _Mapping]] = ..., run: _Optional[_Union[Run, _Mapping]] = ...) -> None: ...

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
    def __init__(self, eval: _Optional[_Union[Eval, _Mapping]] = ..., run: _Optional[_Union[Run, _Mapping]] = ...) -> None: ...

class SubmitEvalResponse(_message.Message):
    __slots__ = ["message"]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    message: str
    def __init__(self, message: _Optional[str] = ...) -> None: ...

class StartTestSuiteRequest(_message.Message):
    __slots__ = ["test_suite"]
    TEST_SUITE_FIELD_NUMBER: _ClassVar[int]
    test_suite: TestSuite
    def __init__(self, test_suite: _Optional[_Union[TestSuite, _Mapping]] = ...) -> None: ...

class StartTestSuiteResponse(_message.Message):
    __slots__ = ["message"]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    message: str
    def __init__(self, message: _Optional[str] = ...) -> None: ...

class EndTestSuiteRequest(_message.Message):
    __slots__ = ["test_suite"]
    TEST_SUITE_FIELD_NUMBER: _ClassVar[int]
    test_suite: TestSuite
    def __init__(self, test_suite: _Optional[_Union[TestSuite, _Mapping]] = ...) -> None: ...

class EndTestSuiteResponse(_message.Message):
    __slots__ = ["message"]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    message: str
    def __init__(self, message: _Optional[str] = ...) -> None: ...
