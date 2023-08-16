from importlib import import_module
import time
from .helpers import BaserunProvider, BaserunStepType, BaserunType

BANNED_CONFIG_KEYS = ['api_base', 'api_key', 'headers', 'organization', 'messages', 'prompt']

DEFAULT_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def monkey_patch_openai(log):
    try:
        openai = import_module("openai")

        original_completion_create = openai.Completion.create
        original_completion_acreate = openai.Completion.acreate
        original_chatcompletion_create = openai.ChatCompletion.create
        original_chatcompletion_acreate = openai.ChatCompletion.acreate

        def patched_completion_create(*args, **kwargs):
            start_time = time.time()
            usage = DEFAULT_USAGE
            output = ""
            try:
                response = original_completion_create(*args, **kwargs)
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

                log(log_entry)

        async def patched_completion_acreate(*args, **kwargs):
            start_time = time.time()
            usage = DEFAULT_USAGE
            output = ""
            try:
                response = await original_completion_acreate(*args, **kwargs)
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

                log(log_entry)

        def patched_chatcompletion_create(*args, **kwargs):
            start_time = time.time()
            usage = DEFAULT_USAGE
            output = ""
            try:
                response = original_chatcompletion_create(*args, **kwargs)
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

                log(log_entry)

        async def patched_chatcompletion_acreate(*args, **kwargs):
            start_time = time.time()
            usage = DEFAULT_USAGE
            output = ""
            try:
                response = await original_chatcompletion_acreate(*args, **kwargs)
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

                log(log_entry)

        openai.Completion.create = patched_completion_create
        openai.Completion.acreate = patched_completion_acreate
        openai.ChatCompletion.create = patched_chatcompletion_create
        openai.ChatCompletion.acreate = patched_chatcompletion_acreate

    except ModuleNotFoundError:
        return
