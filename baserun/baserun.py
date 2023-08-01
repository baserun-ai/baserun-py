import json
import uuid
import os
import requests
import time
import threading
from typing import Union
from urllib.parse import urlparse
import warnings
from .helpers import BaserunProvider, BaserunStepType, BaserunType, get_provider_for_model
from .openai import monkey_patch_openai

_thread_local = threading.local()


class Baserun:
    _initialized = False
    _testExecutions = []
    _api_url = None
    _api_key = None

    @staticmethod
    def init(api_url: str = "https://baserun.ai/api/v1/runs") -> None:
        api_key = os.environ.get('BASERUN_API_KEY')
        if not api_key:
            raise ValueError("Baserun API key is missing. Ensure the BASERUN_API_KEY environment variable is set.")

        if Baserun._initialized:
            warnings.warn("Baserun has already been initialized. Additional calls to init will be ignored.")
            return

        Baserun._api_url = api_url
        Baserun._api_key = api_key
        Baserun._initialized = True

        monkey_patch_openai(Baserun._append_to_buffer)

    @staticmethod
    def test(func):
        def wrapper(*args, **kwargs):
            if not Baserun._initialized:
                return func(*args, **kwargs)

            test_name = func.__name__
            test_inputs = [str(arg) for arg in args] + [str(v) for v in kwargs.values()]

            test_execution_id = str(uuid.uuid4())
            _thread_local.baserun_test_execution_id = test_execution_id
            _thread_local.buffer = []
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
            except Exception as e:
                end_time = time.time()
                Baserun.store_test({
                    'testName': test_name,
                    'testInputs': test_inputs,
                    'id': test_execution_id,
                    'error': str(e),
                    'startTimestamp': start_time,
                    'completionTimestamp': end_time,
                    "steps": _thread_local.buffer
                })
                _thread_local.buffer = []
                del _thread_local.baserun_test_execution_id
                raise e

            end_time = time.time()
            Baserun.store_test({
                'testName': test_name,
                'testInputs': test_inputs,
                'id': test_execution_id,
                'result': str(result) if result is not None else '',
                'startTimestamp': start_time,
                'completionTimestamp': end_time,
                "steps": _thread_local.buffer
            })
            _thread_local.buffer = []
            del _thread_local.baserun_test_execution_id
            return result

        return wrapper

    @staticmethod
    def log(name: str, payload: Union[str, dict]):
        if not Baserun._initialized:
            return

        baserun_test_execution_id = getattr(_thread_local, "baserun_test_execution_id", None)
        if not baserun_test_execution_id:
            warnings.warn("baserun.log was called outside of a Baserun decorated test. The log will be ignored.")
            return

        log_entry = {
            "stepType": BaserunStepType.LOG.name.lower(),
            "name": name,
            "payload": payload,
            "timestamp": time.time(),
        }

        Baserun._append_to_buffer(log_entry)

    @staticmethod
    def log_llm_chat(config: dict, messages: list[dict], output: str, variables: dict[str, str] = None):
        if not Baserun._initialized:
            return

        baserun_test_execution_id = getattr(_thread_local, "baserun_test_execution_id", None)
        if not baserun_test_execution_id:
            warnings.warn(
                "baserun.log_llm_chat was called outside of a Baserun decorated test. The log will be ignored.")
            return

        if 'model' not in config:
            warnings.warn("The 'model' property must be specified in config, falling back to baserun.log")
            Baserun.log(json.dumps({"config": config, "messages": messages, "output": output, "variables": variables}))
            return

        model = config['model']
        provider = get_provider_for_model(model)
        if not isinstance(provider, BaserunProvider):
            warnings.warn(f"The specified model '{model}' is not supported by Baserun, falling back to baserun.log")
            Baserun.log(json.dumps({"config": config, "messages": messages, "output": output, "variables": variables}))
            return

        log_entry = {
            "stepType": BaserunStepType.CUSTOM_LLM.name.lower(),
            "type": BaserunType.CHAT.name.lower(),
            "provider": provider.name.lower(),
            "config": config,
            "messages": [
                {
                    "role": message["role"],
                    "content": message["content"].replace("{", "{{").replace("}", "}}")
                }
                for message in messages
            ],
            "output": output,
            "variables": variables if variables else {},
            "timestamp": time.time(),
        }

        Baserun._append_to_buffer(log_entry)

    @staticmethod
    def log_llm_completion(config: dict[str, any], prompt: str, output: str, variables: dict[str, str] = None):
        if not Baserun._initialized:
            return

        baserun_test_execution_id = getattr(_thread_local, "baserun_test_execution_id", None)
        if not baserun_test_execution_id:
            warnings.warn("baserun.log_llm_completion was called outside of a Baserun decorated test. The log will be ignored.")
            return

        if 'model' not in config:
            warnings.warn("The 'model' property must be specified in config, falling back to baserun.log")
            Baserun.log(json.dumps({"config": config, "prompt": prompt, "output": output, "variables": variables}))
            return

        model = config['model']
        provider = get_provider_for_model(model)
        if not isinstance(provider, BaserunProvider):
            warnings.warn(f"The specified model '{model}' is not supported by Baserun, falling back to baserun.log")
            Baserun.log(json.dumps({"config": config, "prompt": prompt, "output": output, "variables": variables}))
            return

        log_entry = {
            "stepType": BaserunStepType.CUSTOM_LLM.name.lower(),
            "type": BaserunType.COMPLETION.name.lower(),
            "provider": provider.name.lower(),
            "config": config,
            "prompt": prompt.replace("{", "{{").replace("}", "}}"),
            "output": output,
            "variables": variables if variables else {},
            "timestamp": time.time(),
        }

        Baserun._append_to_buffer(log_entry)

    @staticmethod
    def store_test(test_data: dict):
        Baserun._testExecutions.append(test_data)

    @staticmethod
    def flush():
        if not Baserun._initialized:
            warnings.warn("Baserun has not been initialized. No data will be flushed.")
            return

        if not Baserun._testExecutions:
            return

        headers = {"Authorization": f"Bearer {Baserun._api_key}"}
        try:
            response = requests.post(Baserun._api_url, json={"testExecutions": Baserun._testExecutions}, headers=headers)
            response.raise_for_status()

            response_data = response.json()
            test_run_id = response_data['id']
            parsed_url = urlparse(Baserun._api_url)
            url = f"{parsed_url.scheme}://{parsed_url.netloc}/runs/{test_run_id}"
            return url

        except requests.RequestException as e:
            warnings.warn(f"Failed to upload results to Baserun: {str(e)}")

        Baserun._testExecutions = []

    @staticmethod
    def _append_to_buffer(log_entry: dict):
        if not hasattr(_thread_local, 'buffer'):
            _thread_local.buffer = []

        _thread_local.buffer.append(log_entry)
