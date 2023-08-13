from enum import Enum, auto


class BaserunProvider(Enum):
    GOOGLE = auto()
    OPENAI = auto()
    LLAMA = auto()


class BaserunType(Enum):
    CHAT = auto()
    COMPLETION = auto()


class BaserunStepType(Enum):
    LOG = auto()
    AUTO_LLM = auto()
    CUSTOM_LLM = auto()


class TraceType(Enum):
    TEST = auto()
    PRODUCTION = auto()
