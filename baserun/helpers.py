from enum import Enum, auto


class BaserunProvider(Enum):
    ANTHROPIC = auto()
    OPENAI = auto()


class BaserunType(Enum):
    CHAT = auto()
    COMPLETION = auto()


class BaserunStepType(Enum):
    LOG = auto()
    AUTO_LLM = auto()
    CUSTOM_LLM = auto()
