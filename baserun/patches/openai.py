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
            usage = response.get("usage", DEFAULT_USAGE)
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
    def is_streaming(_symbol: str, _args: Sequence[Any], kwargs: Dict[str, Any]):
        return kwargs.get('stream', False)

    @staticmethod
    def collect_streamed_response(symbol: str, response: Any, chunk: Any) -> Any:
        if "ChatCompletion" in symbol:
            if response is None:
                response = {
                    "id": chunk.get("id"),
                    "object": "chat.completion",
                    "created": chunk.get("created"),
                    "model": chunk.get("model"),
                    "choices": [],
                    "usage": DEFAULT_USAGE,
                }

            for new_choice in chunk.get("choices", []):
                new_index = new_choice.get("index", 0)
                new_delta = new_choice.get("delta", {})
                new_content = new_delta.get("content", "")
                new_role = new_delta.get("role", "assistant")
                new_name = new_delta.get("name", None)
                new_function_call = new_delta.get("function_call", None)
                new_finish_reason = new_choice.get("finish_reason")

                for existing_choice in response.get("choices", []):
                    if existing_choice.get("index", -1) == new_index:
                        if new_content:
                            if "content" in existing_choice["message"]:
                                existing_choice["message"]["content"] += new_content
                            else:
                                existing_choice["message"]["content"] = new_content
                        if new_function_call:
                            existing_choice["message"]["function_call"] = new_function_call
                        if new_name:
                            existing_choice["name"] = new_name

                        existing_choice["finish_reason"] = new_finish_reason
                        break
                else:
                    new_choice_obj = {
                        "index": new_index,
                        "message": {
                            "role": new_role
                        },
                        "finish_reason": new_finish_reason
                    }

                    if new_content:
                        new_choice_obj["message"]["content"] = new_content
                    if new_function_call:
                        new_choice_obj["message"]["function_call"] = new_function_call
                    if new_name:
                        new_choice_obj["message"]["name"] = new_name

                    response["choices"].append(new_choice_obj)

            return response
        else:
            if response is None:
                return chunk

            for new_choice in chunk.get("choices", []):
                new_index = new_choice.get("index", 0)
                new_text = new_choice.get("text", "")

                for existing_choice in response.get("choices", []):
                    if existing_choice.get("index", -1) == new_index:
                        existing_choice["text"] += new_text
                        break
                else:
                    response["choices"].append(new_choice)

            return response

    @staticmethod
    def init(log: Callable):
        Patch(
            resolver=OpenAIWrapper.resolver,
            log=log,
            module=openai,
            symbols=list(OpenAIWrapper.original_methods.keys()),
            is_streaming=OpenAIWrapper.is_streaming,
            collect_streamed_response=OpenAIWrapper.collect_streamed_response
        )
