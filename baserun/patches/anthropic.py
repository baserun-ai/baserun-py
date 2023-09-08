import copy
from importlib import import_module
from .patch import Patch
from .constants import DEFAULT_USAGE
from baserun.helpers import BaserunProvider, BaserunStepType, BaserunType
from typing import Any, Callable, Dict, Optional, Sequence

BANNED_CONFIG_KEYS = ['extra-body', 'extra_query', 'extra_headers', 'prompt']


class AnthropicWrapper:
    client = None

    @staticmethod
    def resolver(_symbol: str, _args: Sequence[Any], kwargs: Dict[str, Any], start_time: int, end_time: int, response: Optional[Any], error: Optional[Exception]):
        usage = DEFAULT_USAGE
        output = ""
        baserun_type = BaserunType.COMPLETION
        config = {key: value for key, value in kwargs.items() if key not in BANNED_CONFIG_KEYS}
        prompt = kwargs.get('prompt', "")
        if error:
            output = f"Error: {error}"
        elif response:
            output = response.completion or ""
            prompt_tokens = AnthropicWrapper.client.count_tokens(prompt)
            completion_tokens = AnthropicWrapper.client.count_tokens(output)
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }

        return {
            "stepType": BaserunStepType.AUTO_LLM.name.lower(),
            "type": baserun_type.name.lower(),
            "provider": BaserunProvider.ANTHROPIC.name.lower(),
            "config": config,
            "output": output,
            "prompt": {"content": prompt},
            "startTimestamp": start_time,
            "completionTimestamp": end_time,
            "usage": usage,
        }

    @staticmethod
    def is_streaming(_symbol: str, _args: Sequence[Any], kwargs: Dict[str, Any]):
        return kwargs.get('stream', False)

    @staticmethod
    def collect_streamed_response(_symbol: str, response: Any, chunk: Any) -> Any:
        if not response:
            return copy.deepcopy(chunk)

        response.completion += chunk.completion
        return response

    @staticmethod
    def init(log: Callable):
        try:
            anthropic = import_module("anthropic")

            # Only support anthropic >= 0.3
            if not hasattr(anthropic, "Anthropic"):
                return

            AnthropicWrapper.client = anthropic.Anthropic()
            Patch(
                resolver=AnthropicWrapper.resolver,
                log=log,
                module=anthropic,
                symbols=['resources.Completions.create', 'resources.AsyncCompletions.create'],
                is_streaming=AnthropicWrapper.is_streaming,
                collect_streamed_response=AnthropicWrapper.collect_streamed_response,
            )
        except ModuleNotFoundError:
            return
