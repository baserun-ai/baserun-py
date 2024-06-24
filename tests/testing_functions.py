import argparse
import asyncio
import inspect
import json
import logging
import os
import sys
import traceback
from time import sleep
from typing import List

import openai
from datasets import Dataset
from openai import AsyncOpenAI, NotFoundError, OpenAI
from openai.types import CreateEmbeddingResponse
from openai.types.chat.chat_completion_message import FunctionCall

from baserun import api, evaluate, get_dataset, init, list_datasets, log, submit_dataset, tag
from baserun.integrations.llamaindex import LLamaIndexInstrumentation
from baserun.models.dataset import DatasetMetadata
from baserun.models.evaluators import Correctness, Includes
from baserun.models.experiment import Experiment
from baserun.wrappers.generic import (
    GenericChoice,
    GenericClient,
    GenericCompletion,
    GenericCompletionMessage,
    GenericInputMessage,
)


def openai_chat(prompt="What is the capital of the US?") -> str:
    client = init(OpenAI(), name="openai_chat")
    completion = client.chat.completions.create(
        name="openai_chat completion", model="gpt-4o", messages=[{"role": "user", "content": prompt}]
    )
    return completion.choices[0].message.content


async def openai_chat_async(prompt="What is the capital of the US?") -> str:
    client = init(AsyncOpenAI(), name="openai_chat_async")
    completion = await client.chat.completions.create(
        name="openai_chat_async completion",
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    )
    return completion.choices[0].message.content


def openai_chat_import(prompt="What is the capital of the US?") -> str:
    from baserun import OpenAI

    client = OpenAI()
    completion = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": prompt}])
    return completion.choices[0].message.content


def openai_chat_with_log(prompt="What is the capital of the US?") -> str:
    client = init(OpenAI(), name="openai_chat_with_log")
    completion = client.chat.completions.create(
        name="openai_chat_with_log completion",
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    )
    content = completion.choices[0].message.content
    command = " ".join(sys.argv)
    completion.log("OpenAI Chat Results", metadata={"command": command})
    return content


def openai_chat_unwrapped(prompt="What is the capital of the US?", **kwargs) -> str:
    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": prompt}], **kwargs
    )
    return completion.choices[0].message.content


async def openai_chat_unwrapped_async_streaming(prompt="What is the capital of the US?", **kwargs) -> str:
    client = AsyncOpenAI()
    completion_generator = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        **kwargs,
        stream=True,
    )
    content = ""
    async for chunk in completion_generator:
        if new_content := chunk.choices[0].delta.content:
            content += new_content

    return content


def openai_chat_tools(prompt="Say 'hello world'") -> FunctionCall:
    messages = [{"role": "user", "content": prompt}]

    client = init(OpenAI(), name="openai_chat_tools")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "say",
                "description": "Convert some text to speech",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "The text to speak"},
                    },
                    "required": ["text"],
                },
            },
        }
    ]
    completion = client.chat.completions.create(
        name="openai_chat_tools completion",
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "say"}},
    )
    tool_calls = completion.choices[0].message.tool_calls
    completion.eval("openai_chat_functions.function_call").includes(
        "say",
        actual=json.dumps([tool_call.model_dump() for tool_call in tool_calls]),
    )
    completion.tool_result(tool_calls[0], "success")

    assistant_message = {
        "role": completion.choices[0].message.role,
        "content": completion.choices[0].message.content,
        "tool_calls": [tool_call.model_dump() for tool_call in tool_calls],
    }
    messages.append(assistant_message)
    messages.append(
        {"role": "tool", "content": "wow", "tool_call_id": assistant_message.get("tool_calls")[0].get("id")}
    )
    client.chat.completions.create(
        name="openai_chat_tools tool response",
        model="gpt-4o",
        messages=messages,
        tools=tools,
    )

    return tool_calls


def openai_chat_tools_streaming(prompt="Say 'hello world'") -> FunctionCall:
    from baserun import OpenAI

    client = OpenAI()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "say",
                "description": "Convert some text to speech",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "The text to speak"},
                    },
                    "required": ["text"],
                },
            },
        }
    ]
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "say"}},
    )
    for item in completion:
        print(item)
    return completion


def openai_chat_streaming(prompt="What is the capital of the US?") -> str:
    client = init(OpenAI(), name="openai_chat_streaming")
    completion_generator = client.chat.completions.create(
        name="openai_chat_streaming completion",
        model="gpt-4o",
        stream=True,
        messages=[{"role": "user", "content": prompt}],
    )
    content = ""
    for chunk in completion_generator:
        if new_content := chunk.choices[0].delta.content:
            content += new_content

    return content


async def openai_chat_async_streaming(prompt="What is the capital of the US?") -> str:
    client = init(AsyncOpenAI(), name="openai_chat_async_streaming")
    completion_generator = await client.chat.completions.create(
        name="openai_chat_async_streaming completion",
        model="gpt-4o",
        stream=True,
        messages=[{"role": "user", "content": prompt}],
    )
    content = ""
    async for chunk in completion_generator:
        if new_content := chunk.choices[0].delta.content:
            content += new_content

    return content


def openai_chat_error(prompt="What is the capital of the US?"):
    client = init(OpenAI(), name="openai_chat_error")

    original_api_type = openai.api_type
    try:
        client.chat.completions.create(
            name="openai_chat_error completion",
            model="asdf",
            messages=[{"role": "user", "content": prompt}],
        )
    except NotFoundError as e:
        client.eval("openai_chat_async_streaming.content").includes(e.message, "does not exist")
        return f"Errored with {e} successfully"
    finally:
        openai.api_type = original_api_type


def openai_chat_response_format(prompt="What is the capital of the US?") -> str:
    client = init(OpenAI())
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Respond to the following question in JSON"},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )
    content = completion.choices[0].message.content
    completion.eval("openai_chat.content").includes("Washington")
    return content


def openai_chat_seed(prompt="What is the capital of the US?") -> str:
    client = init(OpenAI())
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        seed=1234,
    )
    content = completion.choices[0].message.content
    completion.eval("openai_chat.content").includes("Washington")
    return content


def openai_embeddings(inpt: str = "What is the capital of the US?") -> CreateEmbeddingResponse:
    client = init(OpenAI())
    res = client.embeddings.create(
        input=inpt,
        model="text-embedding-ada-002",
    )
    return res


TEMPLATES = {
    "Question & Answer": [
        {"role": "system", "content": "Respond to all messages in the form of a limerick"},
        {"role": "user", "content": "{question}"},
    ]
}


def use_template(
    template="Answer this question in the form of a limerick: {question}", question="What is the capital of the US?"
) -> str:
    prompt = [{"role": "user", "content": template.format(question=question)}]

    client = init(OpenAI(), name="use_template")
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=prompt,
        template=template,
        variables={"question": question},
        name="use_template completion",
    )
    content = completion.choices[0].message.content
    completion.eval("openai_chat.content").includes("Washington")
    return content


async def use_template_async(
    template="Answer this question in the form of a limerick: {question}", question="What is the capital of the US?"
) -> str:
    prompt = [{"role": "user", "content": template.format(question=question)}]

    client = init(AsyncOpenAI(), name="use_template async")
    completion = await client.chat.completions.create(
        model="gpt-4o",
        messages=prompt,
        template=template,
        variables={"question": question},
        name="use_template async completion",
    )
    content = completion.choices[0].message.content
    completion.eval("openai_chat.content").includes("Washington")
    return content


def use_sessions(
    prompt="What is the capital of the US?", user_identifier="example@test.com", session_identifier="session-123"
) -> str:
    client = init(OpenAI(), user=user_identifier, session=session_identifier, name="use_sessions")
    completion = client.chat.completions.create(
        name="use_sessions completion",
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    )
    return completion.choices[0].message.content


def create_full_trace(question="What is the capital of the US?") -> str:
    client = init(OpenAI(), name="Full Trace")
    client.log("Answering question for customer", metadata={"customer_tier": "Pro"})

    completion = client.chat.completions.create(
        model="gpt-4-1106-preview",
        temperature=0.3,
        user="epeterson",
        variables={"question": question},
        messages=[{"role": "user", "content": question}],
    )
    content = completion.choices[0].message.content

    completion.feedback(
        name="question_feedback",
        score=0.8,
        metadata={"comment": "This is correct but is too concise"},
    )
    completion.eval("Contains answer").includes("Washington")
    completion.log(name="Extracted content", message=content)
    completion.submit_to_baserun()
    client.output = content
    client.feedback("content", 0.9, metadata={"comment": "This is a great answer"})
    client.eval("Correct answer").includes(expected="Washington")
    return content


def use_input_variables():
    client = init(OpenAI())
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "What is the capital of the US?"}],
        variables={"country": "the US"},
    )
    return completion.choices[0].message.content


def langchain_unwrapped_openai(prompt="What is the capital of the US?") -> str:
    """Requires Langchain, install using `pip install -qU langchain-openai`"""
    from langchain_core.messages import HumanMessage
    from langchain_openai import ChatOpenAI

    from baserun.integrations.langchain import BaserunCallbackManager

    model = ChatOpenAI(model="gpt-4o", callback_manager=BaserunCallbackManager())
    result = model.invoke([HumanMessage(content=prompt)])
    return result.content


def use_tag_function_client():
    client = init(OpenAI(), name="trace to be resumed")
    client.submit_to_baserun()

    submitted_tag = tag("some_key", "some_value", trace_id=client.trace_id)
    return submitted_tag


def use_tag_function_completion():
    client = init(OpenAI(), name="trace to be resumed")
    completion = client.chat.completions.create(
        name="completion to be resumed",
        model="gpt-4o",
        messages=[{"role": "user", "content": "What is the capital of the US?"}],
    )

    submitted_tag = tag("some_key", "some_value", trace_id=client.trace_id, completion_id=completion.completion_id)
    return submitted_tag


def use_log_function_client():
    client = init(OpenAI(), name="trace to be resumed")
    client.submit_to_baserun()

    submitted_tag = log("some_value", trace_id=client.trace_id)
    return submitted_tag


def use_log_function_completion():
    client = init(OpenAI(), name="trace to be resumed")
    completion = client.chat.completions.create(
        name="completion to be resumed",
        model="gpt-4o",
        messages=[{"role": "user", "content": "What is the capital of the US?"}],
    )

    submitted_tag = log("some_value", trace_id=client.trace_id, completion_id=completion.completion_id)
    return submitted_tag


def use_generic_completion():
    client = GenericClient(name="My Traced Client")
    completion = GenericCompletion(
        model="my custom model",
        name="My Completion",
        input_messages=[GenericInputMessage(content="What is the capital of the US?", role="user")],
        choices=[GenericChoice(message=GenericCompletionMessage(content="Washington"))],
        client=client,
        trace_id=client.trace_id,
    )
    completion.submit_to_baserun()


def use_ragas():
    from ragas import evaluate
    from ragas.metrics import answer_correctness, faithfulness

    question = "When was the first super bowl?"

    from baserun import OpenAI

    client = OpenAI(name="Using Ragas")
    completion = client.chat.completions.create(
        model="gpt-4o",
        name="use_ragas completion",
        messages=[{"role": "user", "content": question}],
    )

    answer = completion.choices[0].message.content

    data_samples = {
        "question": [question],
        "answer": [answer],
        "contexts": [
            [
                "The First AFL–NFL World Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles,"
            ],
        ],
        "ground_truth": [
            "The first Super Bowl was held on January 15, 1967",
        ],
    }

    dataset = Dataset.from_dict(data_samples)

    score = evaluate(dataset, metrics=[faithfulness, answer_correctness])
    completion.eval_many(score.scores.data.to_pydict())

    return score


def use_ragas_with_llama_index():
    from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
    from ragas import evaluate
    from ragas.metrics import answer_correctness, faithfulness

    question = "When was the first super bowl?"

    trace_client = LLamaIndexInstrumentation.start().client
    documents = SimpleDirectoryReader("tests/test_data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()
    answer = query_engine.query(
        "I have flour, sugar and butter. What am I missing if I want to bake oatmeal cookies from my recipe?"
    )
    trace_client.output = answer.response

    data_samples = {
        "question": [question],
        "answer": [answer.response],
        "contexts": [
            [
                "Common ingredients for oatmeal cookies include oats, flour, sugar, butter, eggs, baking soda, cinnamon, and raisins."
            ],
        ],
        "ground_truth": [
            "You are missing eggs, vanilla, cinnamon, salt, and oats.",
        ],
    }

    dataset = Dataset.from_dict(data_samples)

    # TODO: Add contexts and ground truth to the evals
    score = evaluate(dataset, metrics=[faithfulness, answer_correctness])
    trace_client.eval_many(score.scores.data.to_pydict())

    return score


async def use_list_datasets() -> List[DatasetMetadata]:
    datasets = await list_datasets()
    return datasets


async def use_get_dataset() -> Dataset:
    dataset = Dataset.from_dict(
        {
            "question": ["When was the first super bowl?"],
            "answer": [
                "The first Super Bowl was held on January 15, 1967. It took place at the Los Angeles Memorial Coliseum in Los Angeles, California."
            ],
            "contexts": [
                [
                    "The First AFL–NFL World Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles,"
                ],
            ],
            "ground_truth": [
                "The first Super Bowl was held on January 15, 1967",
            ],
        }
    )

    submit_dataset(dataset, name="questions", metadata={"description": "Dataset of questions and answers"})

    # Wait for dataset to be posted and persisted
    await asyncio.sleep(2)

    retrieved_dataset = await get_dataset(name="questions")
    return retrieved_dataset


def compile_completions_dataset() -> Dataset:
    dataset = Dataset.from_list([])
    client = init(OpenAI(), name="openai_chat")
    completion = client.chat.completions.create(
        name="openai_chat completion",
        model="gpt-4o",
        messages=[{"role": "user", "content": "What is the capital of the U.S.?"}],
    )
    dataset = completion.add_to_dataset(dataset)
    return json.dumps(dataset.to_list(), indent=2)


async def use_llama_with_chat_engine() -> Dataset:
    from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

    query = "Who won superbowl 31?"

    documents = SimpleDirectoryReader(input_files=["tests/test_data/super_bowl.txt"]).load_data()
    index = VectorStoreIndex.from_documents(documents)
    chat_engine = index.as_chat_engine()

    trace = LLamaIndexInstrumentation.start().client
    trace.name = "Super Bowl"
    trace.variable("query", query)

    answer = chat_engine.query(query)

    trace.output = answer.response
    trace.submit_to_baserun()


async def use_dataset_for_rag_eval() -> Dataset:
    from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

    dataset = Dataset.from_list(
        [
            {
                "input": {
                    "query": "When was the first super bowl?",
                    "context": {
                        "The First AFL–NFL World Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles,"
                    },
                },
                "output": "The first Super Bowl was held on Jan 15, 1967",
                "expected": "1967",
                "metadata": {"name": "Super Bowl 1", "eval_name": "correctness"},
            },
            {
                "input": {
                    "query": "Who won superbowl 31?",
                    "context": {"The Green Bay Packers play in Green Bay, Wisconsin"},
                },
                "output": "The Green Bay Packers won Super Bowl XXXI",
                "expected": "Packers",
                "metadata": {"name": "Super Bowl 31", "eval_name": "correctness"},
            },
        ]
    )
    documents = SimpleDirectoryReader(input_files=["tests/test_data/super_bowl.txt"]).load_data()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()

    for retrieval in dataset.to_list():
        name = retrieval.get("metadata", {}).get("name")
        eval_name = retrieval.get("metadata", {}).get("eval_name", name)

        trace = LLamaIndexInstrumentation.start().client

        query = retrieval.get("input", {}).get("query")
        answer = query_engine.query(query)

        trace.variable("query", query)
        trace.log({"question": query, "answer": answer.response}, name="answer")

        score = 1 if retrieval.get("expected") in answer.response else 0
        trace.eval(eval_name, score)

        trace.output = answer.response
        trace.submit_to_baserun()


async def use_submit_dataset() -> Dataset:
    dataset = Dataset.from_list(
        [
            {
                "input": {"city": "Washington, D.C.", "country": "United States"},
                "expected": "The capital of the {country} is {city}",
                "contexts": ["The United States still exists as a country and the capital has not changed recently."],
                "name": "U.S.",
                "id": "bcc8e116-4f70-4f6d-bb4f-d5892b5db8e1",
            },
            {
                "input": {"city": "London", "country": "United Kingdom"},
                "expected": "The capital of the {country} is {city}",
                "contexts": ["The United Kingdom still exists as a country and the capital has not changed recently."],
                "name": "U.K.",
                "id": "c6efec68-b01d-44f2-b204-54afba7b9ad9",
            },
        ],
    )

    submit_dataset(dataset, name="capital questions", metadata={"description": "Dataset of questions about capitals"})
    return dataset


async def use_dataset_for_online_eval() -> Dataset:
    from baserun import OpenAI

    dataset = await get_dataset(name="capital questions")
    question = "What is the capital of {country}?"

    client = OpenAI(name="Online Eval")
    experiment = Experiment(dataset=dataset, client=client, name="Dataset online eval run")
    for scenario in experiment.scenarios:
        evaluators = [Includes(scenario=scenario, expected="{city}"), Correctness(scenario=scenario, question=question)]

        completion = client.chat.completions.create(
            name=scenario.name,
            model="gpt-4o",
            messages=scenario.format_messages([{"role": "user", "content": question}]),
            variables=scenario.input,
        )
        output = completion.choices[0].message.content
        client.output = output
        scenario.actual = output

        evaluate(evaluators, scenario, completion=completion)

    return dataset


def call_function(functions, function_name: str, parsed_args: argparse.Namespace):
    function_to_call = functions.get(function_name)
    if function_to_call is None:
        function_to_call = {f: globals().get(f) for f in globals()}.get(function_name)

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
    return result


# Allows you to call any of these functions, e.g. python tests/testing_functions.py openai_chat_functions_streaming
if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    openai.api_key = os.environ.get("OPENAI_API_KEY")

    parser = argparse.ArgumentParser(description="Execute a function with a prompt.")
    parser.add_argument("function_to_call", type=str, help="Name of the function to call")
    parser.add_argument("--prompt", type=str, help="Prompt to pass to the function", default=None)

    parsed_args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)

    # Resolve the string function name to the function object
    function_name = parsed_args.function_to_call
    global_variables = {f: globals().get(f) for f in globals()}
    traced_functions = {n: f for n, f in global_variables.items() if callable(f) and f.__name__ == "wrapper"}
    if function_name == "all":
        for name, func in traced_functions.items():
            print(f"===== Calling function {name} =====\n")
            try:
                result = call_function(traced_functions, name, parsed_args)
                print(f"----- {name} result:\n{result}\n-----")
            except Exception:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_exception(exc_type, exc_value, exc_traceback)
    else:
        call_function(traced_functions, function_name, parsed_args)

    sleep(4)
    api.stop_worker()

    sleep(1)
