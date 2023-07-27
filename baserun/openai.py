from importlib import import_module
import time
from .helpers import BaserunProvider, BaserunType


def monkey_patch_openai(log):
    try:
        openai = import_module("openai")

        original_completion_create = openai.Completion.create
        original_chatcompletion_create = openai.ChatCompletion.create

        def patched_completion_create(*args, **kwargs):
            start_time = time.time()
            response = original_completion_create(*args, **kwargs)
            end_time = time.time()

            prompt = kwargs.get('prompt', "")
            config = {key: value for key, value in kwargs.items() if key != 'prompt'}
            usage = response["usage"]
            output = response["choices"][0]["text"]

            log_entry = {
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

            return response

        def patched_chatcompletion_create(*args, **kwargs):
            start_time = time.time()
            response = original_chatcompletion_create(*args, **kwargs)
            end_time = time.time()

            messages = kwargs.get('messages', [])
            config = {key: value for key, value in kwargs.items() if key != 'messages'}
            output = response["choices"][0]["message"]["content"]
            usage = response["usage"]

            log_entry = {
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

            return response

        openai.Completion.create = patched_completion_create
        openai.ChatCompletion.create = patched_chatcompletion_create

    except ModuleNotFoundError:
        return
