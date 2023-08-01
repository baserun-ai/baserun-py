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


SupportedModels = {
    BaserunProvider.OPENAI: [
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-16k",
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4",
        "gpt-4-0613",
        "gpt-4-32k",
        "gpt-4-32k-0613",
        "text-davinci-003",
        "text-curie-001",
        "text-babbage-001",
        "text-ada-001"
    ],
    BaserunProvider.GOOGLE: ["chat-bison@001", "text-bison@001"],
    BaserunProvider.LLAMA: ["llama7b-v2-chat", "llama13b-v2-chat", "llama70b-v2-chat"],
}


def get_provider_for_model(model: str) -> BaserunProvider:
    for provider, models in SupportedModels.items():
        if model in models:
            return provider

    raise ValueError(f"Model '{model}' is not supported by any provider.")
