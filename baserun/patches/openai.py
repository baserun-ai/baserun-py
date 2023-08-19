import openai
from .patch import Patch
from .constants import DEFAULT_USAGE
from baserun.helpers import BaserunProvider, BaserunStepType, BaserunType
from typing import Any, Callable, Dict, Optional, Sequence

BANNED_CONFIG_KEYS = ['api_base', 'api_key', 'headers', 'organization', 'messages', 'prompt']


class OpenAIWrapper:
    original_methods = {
        "Completion.create": openai.Completion.create,
        "Completion.acreate": openai.Completion.acreate,
        "ChatCompletion.create": openai.ChatCompletion.create,
        "ChatCompletion.acreate": openai.ChatCompletion.acreate
    }

    @staticmethod
    def resolver(symbol: str, _args: Sequence[Any], kwargs: Dict[str, Any], start_time: int, end_time: int, response: Optional[Any], error: Optional[Exception]):
        usage = DEFAULT_USAGE
        output = ""
        baserun_type = BaserunType.CHAT if "ChatCompletion" in symbol else BaserunType.COMPLETION
        config = {key: value for key, value in kwargs.items() if key not in BANNED_CONFIG_KEYS}
        if error:
            output = f"Error: {error}"
        elif response:
            usage = response["usage"]
            if baserun_type == BaserunType.CHAT:
                output = response["choices"][0].get("message", "")
            else:
                output = response["choices"][0].get("text", "")

        log_entry = {
            "stepType": BaserunStepType.AUTO_LLM.name.lower(),
            "type": baserun_type.name.lower(),
            "provider": BaserunProvider.OPENAI.name.lower(),
            "config": config,
            "output": output,
            "startTimestamp": start_time,
            "completionTimestamp": end_time,
            "usage": usage,
        }

        if baserun_type == BaserunType.CHAT:
            return {
                **log_entry,
                "messages": kwargs.get('messages', []),
            }
        else:
            return {
                **log_entry,
                "prompt": {"content": kwargs.get('prompt', "")},
            }

    @staticmethod
    def init(log: Callable):
        Patch(
            resolver=OpenAIWrapper.resolver,
            log=log,
            module=openai,
            symbols=list(OpenAIWrapper.original_methods.keys())
        )
