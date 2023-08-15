import openai
import time
from .helpers import BaserunProvider, BaserunStepType, BaserunType
from typing import Dict

BANNED_CONFIG_KEYS = ['api_base', 'api_key', 'headers', 'organization', 'messages', 'prompt']

DEFAULT_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def noop_log(_log_entry: Dict):
    pass


class OpenAIWrapper:
    _log = noop_log
    original_methods = {
        "completion_create": openai.Completion.create,
        "completion_acreate": openai.Completion.acreate,
        "chatcompletion_create": openai.ChatCompletion.create,
        "chatcompletion_acreate": openai.ChatCompletion.acreate
    }

    @staticmethod
    def init(log):
        _log = log

        openai.Completion.create = OpenAIWrapper.patched_completed_create
        openai.Completion.acreate = OpenAIWrapper.patched_completion_acreate
        openai.ChatCompletion.create = OpenAIWrapper.patched_chatcompletion_create
        openai.ChatCompletion.acreate = OpenAIWrapper.patched_completion_acreate

    @staticmethod
    def patched_completed_create(*args, **kwargs):
        start_time = time.time()
        usage = DEFAULT_USAGE
        output = ""
        try:
            response = OpenAIWrapper.original_methods["completion_create"](*args, **kwargs)
            usage = response["usage"]
            output = response["choices"][0]["text"]
            return response
        except Exception as e:
            output = f"Error: {e}"
            raise e
        finally:
            end_time = time.time()

            prompt = kwargs.get('prompt', "")
            config = {key: value for key, value in kwargs.items() if key not in BANNED_CONFIG_KEYS}

            log_entry = {
                "stepType": BaserunStepType.AUTO_LLM.name.lower(),
                "type": BaserunType.COMPLETION.name.lower(),
                "provider": BaserunProvider.OPENAI.name.lower(),
                "config": config,
                "prompt": {"content": prompt},
                "output": output,
                "startTimestamp": start_time,
                "completionTimestamp": end_time,
                "usage": usage,
            }

            OpenAIWrapper._log(log_entry)

    @staticmethod
    async def patched_completion_acreate(*args, **kwargs):
        start_time = time.time()
        usage = DEFAULT_USAGE
        output = ""
        try:
            response = await OpenAIWrapper.original_methods["completion_acreate"](*args, **kwargs)
            usage = response["usage"]
            output = response["choices"][0]["text"]
            return response
        except Exception as e:
            output = f"Error: {e}"
            raise e
        finally:
            end_time = time.time()

            prompt = kwargs.get('prompt', "")
            config = {key: value for key, value in kwargs.items() if key not in BANNED_CONFIG_KEYS}

            log_entry = {
                "stepType": BaserunStepType.AUTO_LLM.name.lower(),
                "type": BaserunType.COMPLETION.name.lower(),
                "provider": BaserunProvider.OPENAI.name.lower(),
                "config": config,
                "prompt": {"content": prompt},
                "output": output,
                "startTimestamp": start_time,
                "completionTimestamp": end_time,
                "usage": usage,
            }

            OpenAIWrapper._log(log_entry)

    @staticmethod
    def patched_chatcompletion_create(*args, **kwargs):
        start_time = time.time()
        usage = DEFAULT_USAGE
        output = ""
        try:
            response = OpenAIWrapper.original_methods["chatcompletion_create"](*args, **kwargs)
            output = response["choices"][0]["message"]
            usage = response["usage"]
            return response
        except Exception as e:
            output = f"Error: {e}"
            raise e
        finally:
            end_time = time.time()

            messages = kwargs.get('messages', [])
            config = {key: value for key, value in kwargs.items() if key not in BANNED_CONFIG_KEYS}

            log_entry = {
                "stepType": BaserunStepType.AUTO_LLM.name.lower(),
                "type": BaserunType.CHAT.name.lower(),
                "provider": BaserunProvider.OPENAI.name.lower(),
                "config": config,
                "messages": messages,
                "output": output,
                "startTimestamp": start_time,
                "completionTimestamp": end_time,
                "usage": usage,
            }

            OpenAIWrapper._log(log_entry)

    @staticmethod
    async def patched_chatcompletion_acreate(*args, **kwargs):
        start_time = time.time()
        usage = DEFAULT_USAGE
        output = ""
        try:
            response = await OpenAIWrapper.original_methods['chatcompletion_acreate'](*args, **kwargs)
            output = response["choices"][0]["message"]
            usage = response["usage"]
            return response
        except Exception as e:
            output = f"Error: {e}"
            raise e
        finally:
            end_time = time.time()

            messages = kwargs.get('messages', [])
            config = {key: value for key, value in kwargs.items() if key not in BANNED_CONFIG_KEYS}

            log_entry = {
                "stepType": BaserunStepType.AUTO_LLM.name.lower(),
                "type": BaserunType.CHAT.name.lower(),
                "provider": BaserunProvider.OPENAI.name.lower(),
                "config": config,
                "messages": messages,
                "output": output,
                "startTimestamp": start_time,
                "completionTimestamp": end_time,
                "usage": usage,
            }

            OpenAIWrapper._log(log_entry)


