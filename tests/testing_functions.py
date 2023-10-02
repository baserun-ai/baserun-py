import argparse
import asyncio
import inspect
import os
from threading import Thread

import openai
from openai import ChatCompletion, Completion

import baserun


@baserun.trace
def openai_chat(prompt="What is the capitol of the US?") -> str:
    completion = ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
    )
    return completion.choices[0]["message"].content


def openai_chat_unwrapped(prompt="What is the capitol of the US?", **kwargs) -> str:
    completion = ChatCompletion.create(
        model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}], **kwargs
    )
    result = completion.choices[0]["message"].content
    print(result)
    return result


@baserun.trace
async def openai_chat_async(prompt="What is the capitol of the US?") -> str:
    completion = await ChatCompletion.acreate(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
    )
    return completion.choices[0]["message"].content


@baserun.trace
def openai_chat_functions(prompt="Say 'hello world'") -> dict[str, str]:
    functions = [
        {
            "name": "say",
            "description": "Convert some text to speech",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The text to speak"},
                },
                "required": ["text"],
            },
        }
    ]
    completion = ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        functions=functions,
        function_call={"name": "say"},
    )
    return completion.choices[0]["message"].function_call


@baserun.trace
def openai_chat_functions_streaming(prompt="Say 'hello world'") -> dict[str, str]:
    functions = [
        {
            "name": "say",
            "description": "Convert some text to speech",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The text to speak"},
                },
                "required": ["text"],
            },
        }
    ]
    completion_generator = ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        functions=functions,
        stream=True,
        function_call={"name": "say"},
    )
    function_name = ""
    function_arguments = ""
    for chunk in completion_generator:
        choice = chunk.choices[0]
        if function_call := choice.delta.get("function_call"):
            function_name += function_call.get("name", "")
            function_arguments += function_call.get("arguments", "")

    return {"name": function_name, "arguments": function_arguments}


@baserun.trace
def openai_chat_streaming(prompt="What is the capitol of the US?") -> str:
    completion_generator = ChatCompletion.create(
        model="gpt-3.5-turbo",
        stream=True,
        messages=[{"role": "user", "content": prompt}],
    )
    content = ""
    for chunk in completion_generator:
        if new_content := chunk.choices[0].delta.get("content"):
            content += new_content

    return content


@baserun.trace
async def openai_chat_async_streaming(prompt="What is the capitol of the US?") -> str:
    completion_generator = await ChatCompletion.acreate(
        model="gpt-3.5-turbo",
        stream=True,
        messages=[{"role": "user", "content": prompt}],
    )
    content = ""
    async for chunk in completion_generator:
        if new_content := chunk.choices[0].delta.get("content"):
            content += new_content

    return content


@baserun.trace
def openai_chat_error(prompt="What is the capitol of the US?"):
    original_api_type = openai.api_type
    try:
        openai.api_type = "somegarbage"
        # Will raise InvalidAPIType
        ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
        )
    finally:
        openai.api_type = original_api_type


@baserun.trace
def traced_fn_error():
    ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "What is the capitol of the US?"}],
    )
    raise ValueError("Something went wrong")


@baserun.trace
def openai_completion(prompt="Human: say this is a test\nAssistant: ") -> str:
    completion = Completion.create(model="text-davinci-003", prompt=prompt)
    return completion.choices[0].text


@baserun.trace
async def openai_completion_async(
    prompt="Human: say this is a test\nAssistant: ",
) -> str:
    completion = await Completion.acreate(model="text-davinci-003", prompt=prompt)
    return completion.choices[0].text


@baserun.trace
def openai_completion_streaming(prompt="Human: say this is a test\nAssistant: ") -> str:
    completion_generator = Completion.create(
        model="text-davinci-003", prompt=prompt, stream=True
    )

    content = ""
    for chunk in completion_generator:
        if new_content := chunk.choices[0].text:
            content += new_content

    return content


@baserun.trace
async def openai_completion_async_streaming(
    prompt="Human: say this is a test\nAssistant: ",
) -> str:
    completion_generator = await Completion.acreate(
        model="text-davinci-003", prompt=prompt, stream=True
    )
    content = ""
    async for chunk in completion_generator:
        if new_content := chunk.choices[0].text:
            content += new_content

    return content


@baserun.trace
def openai_threaded():
    threads = [
        Thread(
            target=baserun.thread_wrapper(openai_chat_unwrapped),
            args=("What is the capitol of the state of Georgia?",),
        ),
        Thread(
            target=baserun.thread_wrapper(openai_chat_unwrapped),
            args=("What is the capitol of the California?",),
            kwargs={"top_p": 0.5},
        ),
        Thread(
            target=baserun.thread_wrapper(openai_chat_unwrapped),
            args=("What is the capitol of the Montana?",),
            kwargs={"temperature": 1},
        ),
    ]
    [t.start() for t in threads]
    [t.join() for t in threads]


# Allows you to call any of these functions, e.g. python tests/testing_functions.py openai_chat_functions_streaming
if __name__ == "__main__":
    from dotenv import load_dotenv
    from baserun import Baserun

    load_dotenv()
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    Baserun.init()

    parser = argparse.ArgumentParser(description="Execute a function with a prompt.")
    parser.add_argument(
        "function_to_call", type=str, help="Name of the function to call"
    )
    parser.add_argument(
        "--prompt", type=str, help="Prompt to pass to the function", default=None
    )

    parsed_args = parser.parse_args()

    # Resolve the string function name to the function object
    function_to_call = globals().get(parsed_args.function_to_call)
    if inspect.iscoroutinefunction(function_to_call):
        if parsed_args.prompt:
            result = asyncio.run(function_to_call(parsed_args.prompt))
        else:
            result = asyncio.run(function_to_call())
    else:
        if parsed_args.prompt:
            result = function_to_call(parsed_args.prompt)
        else:
            result = function_to_call()

    print(result)
